"""Tests for leave-one-out cross-validation.

The whole point of `loo` is that no fold sees the point it is scored against,
so the tests below pin exactly that: what each fold is trained on, that the
prediction is read at the held-out action, and that `fit_gp` is handed the
arguments it was promised.
"""

import numpy as np
import pytest

from pypolar.optimization.gp import BoTorchGP, GPHyperparameters, NoiseModel
from pypolar.optimization.objectives import Objective
from pypolar.performance.loo import loo


LOW  = np.array([0.0, -2.0])
HIGH = np.array([10.0, 2.0])
PEAK = np.array([7.5, 1.0])

NOISE_STD = 0.05


def _grid(n=4):
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(LOW, HIGH)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def _bump(X, center=PEAK, width=3.0):
    return np.exp(-np.sum((X - center) ** 2, axis=1) / width ** 2)


@pytest.fixture
def actions():
    return _grid()


@pytest.fixture
def objective(actions):
    return Objective.from_data(actions=actions, values=_bump(actions),
                               maximize=True, name='reward')


def _fit(objective, noise, hypers):
    """A `fit_gp` of the form `loo` requires, hyperparameters left free."""
    return BoTorchGP(objective, noise=noise, fit_hyperparameters=True)


class TestFolds:

    def test_every_fold_leaves_out_exactly_one_point(self, objective):
        seen = []
        loo(objective, lambda o, n, h: seen.append(o) or _fit(o, n, h),
            NOISE_STD)

        assert len(seen) == objective.ydata.size
        for fold in seen:
            assert fold.ydata.size == objective.ydata.size - 1

    def test_fold_i_is_missing_measurement_i(self, objective):
        seen = []
        loo(objective, lambda o, n, h: seen.append(o) or _fit(o, n, h),
            NOISE_STD)

        for i, fold in enumerate(seen):
            kept = np.delete(objective.ydata, i)
            np.testing.assert_allclose(fold.ydata, kept)
            np.testing.assert_allclose(fold.xdata, np.delete(objective.xdata, i, axis=0))

    def test_the_folds_direction_and_name_are_the_originals(self, actions):
        cost = Objective.from_data(actions=actions, values=_bump(actions),
                                   maximize=False, name='cost')
        seen = []
        loo(cost, lambda o, n, h: seen.append(o) or _fit(o, n, h), NOISE_STD)

        assert all(fold.maximize is False for fold in seen)
        assert all(fold.name == 'cost' for fold in seen)

    def test_it_returns_one_model_per_fold(self, objective):
        _, _, models = loo(objective, _fit, NOISE_STD)
        assert len(models) == objective.ydata.size
        assert all(isinstance(m, BoTorchGP) for m in models)


class TestPredictions:

    def test_shapes_are_one_per_measurement(self, objective):
        mu, std, _ = loo(objective, _fit, NOISE_STD)
        assert mu.shape == std.shape == objective.ydata.shape

    def test_the_prediction_is_the_folds_posterior_at_the_held_out_action(
            self, objective):
        """Not at some other point, and not the training fit of the full GP."""
        mu, std, models = loo(objective, _fit, NOISE_STD)

        for i, gp in enumerate(models):
            expected_mu, expected_std = gp.posterior_at(objective.xdata[i], raw=True)
            assert mu[i] == pytest.approx(expected_mu[0, 0])
            assert std[i] == pytest.approx(expected_std[0, 0])

    def test_it_reports_in_raw_units(self, actions):
        """raw=True, so a prediction of a 1000x objective is 1000x larger."""
        def predicted(scale):
            obj = Objective.from_data(actions=actions, values=scale * _bump(actions),
                                      maximize=True, name='reward')
            return loo(obj, _fit, NOISE_STD)[0]

        np.testing.assert_allclose(predicted(1000.0), 1000.0 * predicted(1.0),
                                   rtol=1e-6)

    def test_a_smooth_objective_is_predicted_well(self, objective):
        """A single bump on a grid is interpolable, so held-out error must be
        small next to the objective's own spread."""
        mu, _, _ = loo(objective, _fit, NOISE_STD)
        rmse = np.sqrt(np.mean((objective.ydata - mu) ** 2))
        assert rmse < 0.5 * objective.ydata.std()

    def test_it_is_not_the_in_sample_fit(self, objective):
        """The held-out prediction differs from the full GP's own training fit;
        equality would mean a fold saw its point."""
        mu, _, _ = loo(objective, _fit, NOISE_STD)
        in_sample = BoTorchGP(objective, noise=NOISE_STD).posterior_at(
            objective.xdata, raw=True)[0][:, 0]
        assert not np.allclose(mu, in_sample)


class TestFitGpArguments:

    def test_noise_and_hypers_reach_fit_gp_unchanged(self, objective):
        noise  = NoiseModel.prior(0.3)
        hypers = GPHyperparameters(lengthscale=0.4, signal_var=2.0, noise_var=0.09)
        seen   = []

        def record(o, n, h):
            seen.append((n, h))
            return BoTorchGP(o, noise=NoiseModel.pinned(np.sqrt(h.noise_var)),
                             fit_hyperparameters=False,
                             length_scale=h.lengthscale, signal_var=h.signal_var)

        loo(objective, record, noise, hypers)
        assert all(n is noise and h is hypers for n, h in seen)

    def test_hypers_defaults_to_none(self, objective):
        seen = []
        loo(objective, lambda o, n, h: seen.append(h) or _fit(o, n, h), NOISE_STD)
        assert all(h is None for h in seen)

    def test_frozen_hyperparameters_reach_every_fold(self, objective):
        """A fit_gp that pins them means every fold's GP reports them back."""
        hypers = GPHyperparameters(lengthscale=0.4, signal_var=2.0, noise_var=0.09)

        def frozen(o, n, h):
            return BoTorchGP(o, noise=NoiseModel.pinned(np.sqrt(h.noise_var)),
                             fit_hyperparameters=False,
                             length_scale=h.lengthscale, signal_var=h.signal_var)

        _, _, models = loo(objective, frozen, None, hypers)
        for gp in models:
            fitted = gp.get_fitted_hyperparameters()
            assert fitted.lengthscale == pytest.approx(0.4)
            assert fitted.signal_var == pytest.approx(2.0)
