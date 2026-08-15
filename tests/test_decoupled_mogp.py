"""Tests for DecoupledMOGP.

The class makes four promises worth pinning down: one independent GP per
objective, a posterior evaluable at *arbitrary* actions (not just measured
ones), a shared normalized action frame that raw actions are mapped through,
and a maximization space that hides the sign of a minimized objective. Each
test below states one of those.
"""

from dataclasses import fields

import numpy as np
import pytest
import torch

from pypolar.optimization.gp import (
    LENGTH_SCALE,
    SIGNAL_VAR,
    DecoupledMOGP,
    NoiseModel,
)
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

# Raw action box, deliberately neither unit nor centered, so a test that
# forgets the normalizing xtransform fails visibly.
LOW  = np.array([0.0, -2.0])
HIGH = np.array([10.0, 2.0])

PEAK   = np.array([7.5, 1.0])    # argmax of the maximized objective, raw units
VALLEY = np.array([2.5, -1.0])   # argmin of the minimized objective, raw units

GRID_POINTS  = 8                        # measurements per action dimension
GRID_SPACING = 1 / (GRID_POINTS - 1)    # between measurements, normalized frame


def _grid(n=GRID_POINTS):
    """An n x n grid over the raw action box."""
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(LOW, HIGH)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def _bump(X, center, width=3.0):
    """A smooth unimodal bump, peaked at center."""
    return np.exp(-np.sum((X - center) ** 2, axis=1) / width ** 2)


@pytest.fixture
def actions():
    return _grid()


@pytest.fixture
def objectives(actions):
    """One maximized and one minimized objective, each with a known optimum.

    'reward' peaks at PEAK and is maximized; 'cost' dips at VALLEY and is
    minimized. Both are measured at the same actions here, but nothing in
    DecoupledMOGP requires that.
    """
    objs = DecoupledObjectives.from_empty()
    objs.add_objective(Objective.from_data(
        actions=actions, values=_bump(actions, PEAK), maximize=True, name='reward'
    ))
    objs.add_objective(Objective.from_data(
        actions=actions, values=-_bump(actions, VALLEY), maximize=False, name='cost'
    ))
    return objs


@pytest.fixture
def mogp(objectives):
    """Hyperparameters fitted, as the pilot script uses it."""
    return DecoupledMOGP(objectives, fit_hyperparameters=True)


@pytest.fixture
def interpolating_mogp(objectives):
    """Near-noiseless, so the posterior must pass through the measurements."""
    return DecoupledMOGP(objectives, fit_hyperparameters=False, noise=1e-4)


# ---- one GP per objective --------------------------------------------------

class TestModelStructure:

    def test_one_submodel_per_objective(self, mogp, objectives):
        assert len(mogp.model.models) == len(objectives)

    def test_submodels_hold_their_own_objective_data(self, mogp, objectives):
        for i, gp in enumerate(mogp.model.models):
            assert gp.train_inputs[0].shape[0] == len(objectives.feedback(i))

    def test_objectives_may_have_different_numbers_of_points(self, actions):
        """Decoupled means each objective carries its own measurements."""
        objs = DecoupledObjectives.from_empty()
        objs.add_objective(Objective.from_data(
            actions=actions, values=_bump(actions, PEAK), maximize=True, name='a'
        ))
        objs.add_objective(Objective.from_data(
            actions=actions[:10], values=_bump(actions[:10], VALLEY),
            maximize=True, name='b'
        ))
        model = DecoupledMOGP(objs, fit_hyperparameters=False)

        assert model.model.models[0].train_inputs[0].shape[0] == len(actions)
        assert model.model.models[1].train_inputs[0].shape[0] == 10

    def test_fitting_moves_the_lengthscales_off_their_prior(self, objectives):
        fixed  = DecoupledMOGP(objectives, fit_hyperparameters=False)
        fitted = DecoupledMOGP(objectives, fit_hyperparameters=True)

        def lengthscales(m):
            return np.concatenate([
                gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
                for gp in m.model.models
            ])

        assert np.allclose(lengthscales(fixed), LENGTH_SCALE)
        assert not np.allclose(lengthscales(fitted), LENGTH_SCALE)


# ---- posterior_at ----------------------------------------------------------

class TestPosteriorAt:

    def test_shapes_are_n_by_num_objectives(self, mogp, objectives):
        X = np.array([[1.0, 0.0], [5.0, 1.5], [9.0, -1.0]])
        mu, std = mogp.posterior_at(X)
        assert mu.shape == (3, len(objectives))
        assert std.shape == (3, len(objectives))

    def test_a_single_action_is_promoted_to_one_row(self, mogp, objectives):
        mu, std = mogp.posterior_at(np.array([5.0, 0.0]))
        assert mu.shape == (1, len(objectives))
        assert std.shape == (1, len(objectives))

    def test_standard_deviations_are_positive(self, mogp):
        _, std = mogp.posterior_at(_grid(5))
        assert np.all(std > 0)

    def test_evaluates_off_the_measured_actions(self, mogp, actions):
        """Nothing snaps to a grid: midpoints between measurements are fine."""
        midpoints = 0.5 * (actions[:-1] + actions[1:])
        mu, _ = mogp.posterior_at(midpoints)
        assert mu.shape == (len(actions) - 1, 2)
        assert np.all(np.isfinite(mu))

    def test_chunking_does_not_change_the_answer(self, mogp, actions):
        whole = mogp.posterior_at(actions, chunk=4096)
        split = mogp.posterior_at(actions, chunk=7)
        np.testing.assert_allclose(whole[0], split[0])
        np.testing.assert_allclose(whole[1], split[1])

    def test_normalized_flag_selects_the_action_frame(self, mogp, objectives):
        """Passing raw actions equals passing them pre-normalized by hand."""
        raw = np.array([[1.0, 0.0], [5.0, 1.5]])
        from_raw  = mogp.posterior_at(raw, normalized=False)
        from_norm = mogp.posterior_at(objectives.xtransform(raw), normalized=True)
        np.testing.assert_allclose(from_raw[0], from_norm[0])
        np.testing.assert_allclose(from_raw[1], from_norm[1])

    def test_raw_actions_are_not_silently_treated_as_normalized(self, mogp):
        """The raw box is not [0, 1]^2, so the two frames must disagree."""
        raw = np.array([[7.5, 1.0]])
        assert not np.allclose(mogp.posterior_at(raw, normalized=False)[0],
                               mogp.posterior_at(raw, normalized=True)[0])

    def test_near_noiseless_posterior_passes_through_the_data(
            self, interpolating_mogp, objectives, actions):
        """With negligible noise the GP interpolates its measurements."""
        mu, _ = interpolating_mogp.posterior_at(actions)
        for i in range(len(objectives)):
            observed = objectives.feedback(i)
            spread = observed.std()
            np.testing.assert_allclose(mu[:, i], observed, atol=0.02 * spread)

    def test_uncertainty_is_lower_at_measured_actions(self, mogp, actions):
        """A measured corner is better known than a point outside the data."""
        _, at_data = mogp.posterior_at(actions[:1])
        _, off_data = mogp.posterior_at(np.array([[20.0, 8.0]]))
        assert np.all(at_data[0] < off_data[0])

    def test_reports_maximization_space(self, mogp, objectives):
        """Both objectives read larger-is-better, matching feedback()."""
        mu, _ = mogp.posterior_at(np.vstack([PEAK, VALLEY]))
        # PEAK is best for 'reward' (max), VALLEY is best for 'cost' (min)
        assert mu[0, 0] > mu[1, 0]
        assert mu[1, 1] > mu[0, 1]


class TestPosteriorAtRawUnits:
    """`raw=True` reports in the units the measurements were taken in, which is
    where a mean and a standard deviation stop transforming the same way."""

    def test_the_minimized_objective_reads_lower_is_better(self, mogp):
        mu, _ = mogp.posterior_at(np.vstack([PEAK, VALLEY]), raw=True)
        # 'cost' is back in its own units, so VALLEY is now the *smaller* value
        assert mu[1, 1] < mu[0, 1]
        assert mu[0, 0] > mu[1, 0]      # 'reward' is maximized, so unchanged

    def test_it_lands_in_the_range_of_the_measurements(self, mogp, objectives):
        mu, _ = mogp.posterior_at(_grid(5), raw=True)
        for i, obj in enumerate(objectives.objectives):
            assert mu[:, i].min() >= obj.ydata.min() - 0.5 * obj.ydata.std()
            assert mu[:, i].max() <= obj.ydata.max() + 0.5 * obj.ydata.std()

    def test_standard_deviations_stay_positive(self, mogp):
        _, std = mogp.posterior_at(_grid(5), raw=True)
        assert np.all(std > 0)

    def test_a_standard_deviation_is_scaled_not_shifted(self, mogp, objectives):
        """The whole point of inv_scale: inv() would add the mean back on."""
        _, std = mogp.posterior_at(_grid(5))
        _, std_raw = mogp.posterior_at(_grid(5), raw=True)
        for i, obj in enumerate(objectives.objectives):
            np.testing.assert_allclose(std_raw[:, i], std[:, i] * obj.ydata.std())

    def test_it_matches_converting_by_hand(self, mogp, objectives):
        mu, std = mogp.posterior_at(_grid(4))
        by_hand = objectives.to_raw(mu, std)
        got = mogp.posterior_at(_grid(4), raw=True)
        np.testing.assert_allclose(got[0], by_hand[0])
        np.testing.assert_allclose(got[1], by_hand[1])


# ---- sample_paths ----------------------------------------------------------

class TestSamplePaths:
    """Draws from the *joint* posterior, where `posterior_at` reports only the
    marginals. Joint over the actions, but still independent across objectives,
    since that independence is what 'decoupled' claims.

    Every test seeds torch first, since the draws come from its global generator.
    """

    def test_shape_is_q_by_n_by_m(self, mogp, objectives):
        paths = mogp.sample_paths(_grid(3), num_paths=5)
        assert paths.shape == (5, 9, len(objectives))

    def test_a_single_action_is_promoted_to_one_row(self, mogp, objectives):
        paths = mogp.sample_paths(np.array([5.0, 0.0]), num_paths=4)
        assert paths.shape == (4, 1, len(objectives))

    def test_marginals_match_posterior_at(self, mogp):
        """Averaged over enough draws, the paths reproduce the mean and standard
        deviation `posterior_at` reports, for every objective."""
        X = _grid(3)
        torch.manual_seed(0)
        paths   = mogp.sample_paths(X, num_paths=4000)
        mu, std = mogp.posterior_at(X)
        np.testing.assert_allclose(paths.mean(axis=0), mu, atol=0.05)
        np.testing.assert_allclose(paths.std(axis=0), std, rtol=0.1)

    def test_draws_are_joint_over_actions(self, mogp):
        """Two actions a hair apart are effectively the same point of the
        function, so their values must move together across draws."""
        X = np.array([[5.0, 0.0], [5.0 + 1e-3, 0.0]])
        torch.manual_seed(0)
        paths = mogp.sample_paths(X, num_paths=200)
        for i in range(paths.shape[2]):
            assert np.corrcoef(paths[:, 0, i], paths[:, 1, i])[0, 1] > 0.99

    def test_draws_are_independent_across_objectives(self, mogp):
        """The objectives are decoupled, so there is no cross-objective
        covariance for a draw to carry."""
        torch.manual_seed(0)
        paths = mogp.sample_paths(np.array([[5.0, 0.0]]), num_paths=2000)[:, 0, :]
        assert abs(np.corrcoef(paths[:, 0], paths[:, 1])[0, 1]) < 0.1

    def test_normalized_flag_selects_the_action_frame(self, mogp, objectives):
        """Passing raw actions equals passing them pre-normalized by hand."""
        raw = np.array([[1.0, 0.0], [5.0, 1.5]])
        torch.manual_seed(0)
        from_raw = mogp.sample_paths(raw, num_paths=3)
        torch.manual_seed(0)
        from_norm = mogp.sample_paths(objectives.xtransform(raw), num_paths=3,
                                      normalized=True)
        np.testing.assert_allclose(from_raw, from_norm)

    def test_raw_units_match_converting_by_hand(self, mogp, objectives):
        torch.manual_seed(0)
        standard = mogp.sample_paths(_grid(3), num_paths=3)
        torch.manual_seed(0)
        got = mogp.sample_paths(_grid(3), num_paths=3, raw=True)
        np.testing.assert_allclose(got, objectives.to_raw(standard))

    def test_raw_restores_the_sign_of_the_minimized_objective(self, mogp):
        """'cost' reads larger-is-better at VALLEY in maximization space, and
        lower-is-better there in its own units. 'reward' is unchanged."""
        X = np.vstack([PEAK, VALLEY])
        torch.manual_seed(0)
        standard = mogp.sample_paths(X, num_paths=400).mean(axis=0)
        torch.manual_seed(0)
        raw = mogp.sample_paths(X, num_paths=400, raw=True).mean(axis=0)
        assert standard[1, 1] > standard[0, 1]
        assert raw[1, 1] < raw[0, 1]
        assert raw[0, 0] > raw[1, 0]

    def test_a_near_noiseless_mogp_pins_its_paths_to_the_data(
            self, interpolating_mogp, actions):
        """With negligible noise every draw passes through the measurements, so
        the paths have almost no spread there."""
        torch.manual_seed(0)
        paths = interpolating_mogp.sample_paths(actions, num_paths=50)
        assert paths.std(axis=0).max() < 0.02


# ---- best_actions ----------------------------------------------------------

class TestBestActions:

    def test_shapes(self, mogp, objectives, actions):
        best, mu, std = mogp.best_actions(num_restarts=4, raw_samples=128)
        assert best.shape == (len(objectives), actions.shape[1])
        assert mu.shape == (len(objectives),)
        assert std.shape == (len(objectives),)

    def test_returns_raw_units_inside_the_action_box(self, mogp):
        best, _, _ = mogp.best_actions(num_restarts=4, raw_samples=128)
        assert np.all(best >= LOW - 1e-9)
        assert np.all(best <= HIGH + 1e-9)

    def test_recovers_the_maximized_objectives_peak(self, mogp, objectives):
        best, _, _ = mogp.best_actions(num_restarts=8, raw_samples=256)
        # compared in the normalized frame, where one tolerance covers both
        # dimensions; GRID_SPACING is the distance between measured actions
        np.testing.assert_allclose(objectives.xtransform(best)[0],
                                   objectives.xtransform(PEAK)[0],
                                   atol=GRID_SPACING)

    def test_recovers_the_minimized_objectives_valley(self, mogp, objectives):
        """A minimized objective is handled by the sign in ytransform, so the
        argmax of the posterior is the *lowest* raw cost."""
        best, _, _ = mogp.best_actions(num_restarts=8, raw_samples=256)
        np.testing.assert_allclose(objectives.xtransform(best)[1],
                                   objectives.xtransform(VALLEY)[0],
                                   atol=GRID_SPACING)

    def test_values_match_the_posterior_at_those_actions(self, mogp):
        best, mu, std = mogp.best_actions(num_restarts=4, raw_samples=128)
        mu_full, std_full = mogp.posterior_at(best)
        np.testing.assert_allclose(mu, np.diag(mu_full))
        np.testing.assert_allclose(std, np.diag(std_full))

    def test_beats_every_measured_action(self, mogp, actions):
        """Optimizing the continuous box can only match or beat the grid."""
        best, mu, _ = mogp.best_actions(num_restarts=8, raw_samples=256)
        on_grid = mogp.posterior_at(actions)[0].max(axis=0)
        assert np.all(mu >= on_grid - 1e-6)

    def test_raw_units_leave_the_actions_alone(self, mogp):
        """Only the values change; the actions are in raw units either way."""
        best, _, _ = mogp.best_actions(num_restarts=4, raw_samples=128)
        best_raw, _, _ = mogp.best_actions(num_restarts=4, raw_samples=128, raw=True)
        np.testing.assert_allclose(best, best_raw, atol=1e-6)

    def test_raw_values_match_the_posterior_at_those_actions(self, mogp):
        best, mu, std = mogp.best_actions(num_restarts=4, raw_samples=128, raw=True)
        mu_full, std_full = mogp.posterior_at(best, raw=True)
        np.testing.assert_allclose(mu, np.diag(mu_full))
        np.testing.assert_allclose(std, np.diag(std_full))


# ---- update_feedback -------------------------------------------------------

class TestUpdateFeedback:

    def test_refits_against_new_measurements(self, objectives, actions):
        model = DecoupledMOGP(objectives, fit_hyperparameters=False, noise=1e-4)
        probe = np.array([[5.0, 0.0]])
        before = model.posterior_at(probe)[0].copy()

        # a measurement far from the current posterior, at the probe itself
        objectives.add_point('reward', probe, np.array([5.0]))
        model.update_feedback(objectives)

        assert not np.allclose(before[:, 0], model.posterior_at(probe)[0][:, 0])

    def test_returns_the_rebuilt_model_in_eval_mode(self, mogp, objectives):
        returned = mogp.update_feedback(objectives)
        assert returned is mogp.model
        assert not returned.training


# ---- hyperparameters -------------------------------------------------------

FIELDS = {'lengthscale', 'signal_var', 'noise_var', 'standardize_scale'}

NOISE_STD = 0.05    # the DecoupledMOGP default


class TestGetFittedHyperparameters:
    """One `GPHyperparameters` per objective, because the objectives are
    decoupled: each GP has its own kernel and nothing is tied across them."""

    def test_one_entry_per_objective_with_the_four_hyperparameters(
            self, mogp, objectives):
        hypers = mogp.get_fitted_hyperparameters()
        assert len(hypers) == len(objectives)
        assert all({f.name for f in fields(h)} == FIELDS for h in hypers)

    def test_it_is_in_objective_order(self, mogp):
        """Entry i belongs to sub-model i, which is objective i."""
        for h, gp in zip(mogp.get_fitted_hyperparameters(), mogp.model.models):
            np.testing.assert_allclose(
                h.lengthscale,
                gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
            )
            assert h.signal_var == pytest.approx(gp.covar_module.outputscale.item())

    def test_one_lengthscale_per_action_dimension(self, mogp, actions):
        for h in mogp.get_fitted_hyperparameters():
            assert h.lengthscale.shape == (actions.shape[1],)

    def test_nothing_torch_escapes(self, mogp):
        for h in mogp.get_fitted_hyperparameters():
            assert isinstance(h.lengthscale, np.ndarray)
            assert all(isinstance(getattr(h, name), float)
                       for name in FIELDS - {'lengthscale'})

    def test_the_noise_is_the_squared_fraction_for_every_objective(self, mogp):
        """noise_std is a fraction of each objective's own spread, so after
        Standardize every sub-model reports the same variance."""
        for h in mogp.get_fitted_hyperparameters():
            assert h.noise_var == pytest.approx(NOISE_STD ** 2)

    def test_an_unfitted_model_reports_the_starting_values(self, objectives):
        mogp = DecoupledMOGP(objectives, fit_hyperparameters=False)
        for h in mogp.get_fitted_hyperparameters():
            assert h.lengthscale == pytest.approx(LENGTH_SCALE)
            assert h.signal_var == pytest.approx(SIGNAL_VAR)

    def test_each_objective_gets_its_own_lengthscales(self, actions):
        """A shared fit would be a bug. The two objectives here differ only in
        smoothness, so the smoother one must come out with the longer
        lengthscales in every action dimension.

        The `objectives` fixture cannot show this: PEAK and VALLEY sit at
        [0.75, 0.75] and [0.25, 0.25] in the normalized frame, and the sign flip
        on the minimized cost turns -bump back into +bump, so the two data sets
        are reflections of each other through the center of the box. A
        reflection leaves every RBF distance unchanged, which makes the two
        marginal likelihoods the same function of the lengthscales."""
        objs = DecoupledObjectives.from_empty()
        objs.add_objective(Objective.from_data(
            actions=actions, values=_bump(actions, PEAK, width=6.0),
            maximize=True, name='wide'
        ))
        objs.add_objective(Objective.from_data(
            actions=actions, values=_bump(actions, PEAK, width=1.5),
            maximize=True, name='narrow'
        ))
        wide, narrow = DecoupledMOGP(objs, fit_hyperparameters=True
                                     ).get_fitted_hyperparameters()

        assert np.all(wide.lengthscale > narrow.lengthscale)

    def test_it_tracks_a_refit_after_new_measurements(self, objectives):
        mogp = DecoupledMOGP(objectives, fit_hyperparameters=True)
        before = [h.lengthscale.copy() for h in mogp.get_fitted_hyperparameters()]

        # a measurement contradicting the bump, on 'reward' only
        objectives.add_point('reward', np.array([[5.0, 0.0]]), np.array([5.0]))
        mogp.update_feedback(objectives)
        after = mogp.get_fitted_hyperparameters()

        assert not np.allclose(before[0], after[0].lengthscale)
        np.testing.assert_allclose(before[1], after[1].lengthscale)


# ---- fitted noise and bounded lengthscales ---------------------------------

class TestFittedNoise:
    """noise=None fits the noise instead of pinning it. The objectives are
    decoupled, so each sub-model fits its own."""

    def test_each_objective_fits_its_own_noise(self, actions):
        """A shared noise would be a bug: only 'noisy' has anything to explain."""
        rng = np.random.default_rng(0)
        clean = _bump(actions, PEAK)
        objs = DecoupledObjectives.from_empty()
        objs.add_objective(Objective.from_data(
            actions=actions, values=clean, maximize=True, name='clean'
        ))
        objs.add_objective(Objective.from_data(
            actions=actions,
            values=clean + rng.normal(0.0, 0.5 * clean.std(), clean.shape),
            maximize=True, name='noisy'
        ))
        mogp = DecoupledMOGP(objs, fit_hyperparameters=True, noise=None)
        clean_h, noisy_h = mogp.get_fitted_hyperparameters()

        assert noisy_h.noise_var > clean_h.noise_var

    def test_it_refuses_to_leave_the_noise_undetermined(self, objectives):
        with pytest.raises(ValueError, match='fit_hyperparameters'):
            DecoupledMOGP(objectives, fit_hyperparameters=False, noise=None)

    def test_the_posterior_still_comes_back_in_both_spaces(self, objectives, actions):
        """Fitting the noise changes the likelihood, not the return contract."""
        mogp = DecoupledMOGP(objectives, fit_hyperparameters=True, noise=None)
        mu, std = mogp.posterior_at(actions)
        assert mu.shape == std.shape == (len(actions), len(objectives))
        assert np.all(std > 0)


class TestConstructorHyperparameters:
    """The starting hyperparameters and the lengthscale bound are per-instance
    arguments, shared by every sub-model."""

    @staticmethod
    def _bounded(objectives):
        return DecoupledMOGP(objectives, fit_hyperparameters=True,
                             min_length_scale=0.5)

    def test_the_default_is_no_bound(self, mogp):
        assert mogp.min_length_scale is None

    def test_every_sub_model_respects_it(self, objectives):
        for h in self._bounded(objectives).get_fitted_hyperparameters():
            assert np.all(h.lengthscale >= 0.5)

    def test_it_survives_a_refit(self, objectives):
        mogp = self._bounded(objectives)
        objectives.add_point('reward', np.array([[5.0, 0.0]]), np.array([5.0]))
        mogp.update_feedback(objectives)
        for h in mogp.get_fitted_hyperparameters():
            assert np.all(h.lengthscale >= 0.5)

    def test_the_starting_hyperparameters_are_arguments_too(self, objectives):
        mogp = DecoupledMOGP(objectives, fit_hyperparameters=False,
                             length_scale=0.7, signal_var=3.0)
        for h in mogp.get_fitted_hyperparameters():
            assert h.lengthscale == pytest.approx(0.7)
            assert h.signal_var == pytest.approx(3.0)

    def test_the_defaults_are_the_module_constants(self, objectives):
        mogp = DecoupledMOGP(objectives, fit_hyperparameters=False)
        for h in mogp.get_fitted_hyperparameters():
            assert h.lengthscale == pytest.approx(LENGTH_SCALE)
            assert h.signal_var == pytest.approx(SIGNAL_VAR)


class TestNoiseModelAcrossObjectives:
    """One NoiseModel is shared, but each sub-model fits its own value from its
    own term of the summed likelihood."""

    def test_a_prior_applies_to_every_sub_model(self, objectives):
        mogp = DecoupledMOGP(objectives, noise=NoiseModel.prior(0.3))
        for gp in mogp.model.models:
            assert 'likelihood.noise_covar.noise_prior' in \
                [p[0] for p in gp.named_priors()]

    def test_the_prior_still_lets_them_differ(self, actions):
        """Shared regularization, not a shared value: only 'noisy' has anything
        to explain, so it must land higher."""
        rng = np.random.default_rng(0)
        clean = _bump(actions, PEAK)
        objs = DecoupledObjectives.from_empty()
        objs.add_objective(Objective.from_data(
            actions=actions, values=clean, maximize=True, name='clean'))
        objs.add_objective(Objective.from_data(
            actions=actions,
            values=clean + rng.normal(0.0, 0.5 * clean.std(), clean.shape),
            maximize=True, name='noisy'))
        clean_h, noisy_h = DecoupledMOGP(
            objs, noise=NoiseModel.prior(0.3)).get_fitted_hyperparameters()

        assert noisy_h.noise_var > clean_h.noise_var
