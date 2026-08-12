"""Tests for build_botorch_gp and BoTorchGP.

`build_botorch_gp` is the single place a `SingleTaskGP` is configured, so its
tests pin the configuration itself: ARD kernel, zero mean, and a noise level
expressed as a fraction of the data's own spread rather than a raw magnitude.

`BoTorchGP` is the single-objective counterpart of `DecoupledMOGP`: one
`Objective` instead of a collection, so the action frame is that objective's own
normalization rather than a shared one. The promises are otherwise the same -- a
posterior evaluable at arbitrary actions, and a maximization space that hides
the sign of a minimized objective -- and the tests are written to match.
"""

import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ZeroMean

from pypolar.optimization.gp import BoTorchGP, build_botorch_gp, gp_hyperparameters
from pypolar.optimization.objectives import Objective


# ---- fixtures --------------------------------------------------------------

# Raw action box, deliberately neither unit nor centered, so a test that
# forgets the normalizing xtransform fails visibly.
LOW  = np.array([0.0, -2.0])
HIGH = np.array([10.0, 2.0])

PEAK   = np.array([7.5, 1.0])    # argmax of the maximized objective, raw units
VALLEY = np.array([2.5, -1.0])   # argmin of the minimized objective, raw units

GRID_POINTS  = 8                        # measurements per action dimension
GRID_SPACING = 1 / (GRID_POINTS - 1)    # between measurements, normalized frame

NOISE_STD = 0.05


def _grid(n=GRID_POINTS):
    """An n x n grid over the raw action box."""
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(LOW, HIGH)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def _bump(X, center, width=3.0):
    """A smooth unimodal bump, peaked at center."""
    return np.exp(-np.sum((X - center) ** 2, axis=1) / width ** 2)


def _targets(model):
    """The values a model was conditioned on, with Standardize undone.

    `train_targets` holds the outcome-transformed values, which are not quite
    `standard_y`: Standardize divides by the sample standard deviation (ddof=1)
    where `AffineTransform.make_standardized` uses the population one (ddof=0).
    Untransforming is what puts them back on a common footing.
    """
    Y, _ = model.outcome_transform.untransform(model.train_targets.reshape(-1, 1))
    return Y.detach().numpy().ravel()


@pytest.fixture
def actions():
    return _grid()


@pytest.fixture
def reward(actions):
    """A maximized objective, peaked at PEAK."""
    return Objective.from_data(
        actions=actions, values=_bump(actions, PEAK), maximize=True, name='reward'
    )


@pytest.fixture
def cost(actions):
    """A minimized objective, at its lowest raw value at VALLEY."""
    return Objective.from_data(
        actions=actions, values=-_bump(actions, VALLEY), maximize=False, name='cost'
    )


@pytest.fixture
def gp(reward):
    """Hyperparameters fitted, as the pilot script uses it."""
    return BoTorchGP(reward, noise_std=NOISE_STD, fit_hyperparameters=True)


@pytest.fixture
def cost_gp(cost):
    return BoTorchGP(cost, noise_std=NOISE_STD, fit_hyperparameters=True)


@pytest.fixture
def interpolating_gp(reward):
    """Near-noiseless, so the posterior must pass through the measurements."""
    return BoTorchGP(reward, noise_std=1e-4, fit_hyperparameters=False)


# ---- build_botorch_gp ------------------------------------------------------

class TestBuildBotorchGpStructure:
    """The kernel, mean and likelihood the rest of the package assumes."""

    @pytest.fixture
    def built(self, actions):
        return build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise_std=NOISE_STD, signal_var=1.0, length_scale=0.2
        )

    def test_it_is_a_single_task_gp(self, built):
        assert isinstance(built, SingleTaskGP)

    def test_the_training_data_is_attached(self, built, actions):
        assert built.train_inputs[0].shape == (len(actions), actions.shape[1])
        assert built.train_targets.shape == (len(actions),)

    def test_everything_is_float64(self, built):
        # the package runs in double precision throughout
        assert built.train_inputs[0].dtype == torch.float64
        assert built.covar_module.base_kernel.lengthscale.dtype == torch.float64

    def test_one_lengthscale_per_action_dimension(self, built, actions):
        """ARD: each action dimension gets its own relevance."""
        assert built.covar_module.base_kernel.lengthscale.shape[-1] == actions.shape[1]

    def test_a_one_dimensional_action_gets_one_lengthscale(self, actions):
        built = build_botorch_gp(
            actions=actions[:, :1], values=_bump(actions, PEAK),
            noise_std=NOISE_STD, signal_var=1.0, length_scale=0.2
        )
        assert built.covar_module.base_kernel.lengthscale.shape[-1] == 1

    def test_the_starting_hyperparameters_are_the_requested_ones(self, actions):
        built = build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise_std=NOISE_STD, signal_var=3.0, length_scale=0.7
        )
        assert built.covar_module.base_kernel.lengthscale.detach().numpy() == \
            pytest.approx(0.7)
        assert built.covar_module.outputscale.item() == pytest.approx(3.0)

    def test_the_prior_mean_is_zero(self, built):
        """Correct rather than assumed: Standardize centers the targets, so the
        prior mean of the standardized values genuinely is zero."""
        assert isinstance(built.mean_module, ZeroMean)
        assert isinstance(built.outcome_transform, Standardize)

    def test_the_noise_is_fixed_not_fitted(self, built):
        """train_Yvar is what selects FixedNoiseGaussianLikelihood, so a later
        fit_gpytorch_mll call only ever moves the kernel."""
        assert isinstance(built.likelihood, FixedNoiseGaussianLikelihood)


class TestBuildBotorchGpNoise:
    """noise_std is a fraction of the data's own standard deviation. Standardize
    divides train_Yvar by the same variance it divides train_Y by, so
    pre-multiplying by the spread is what makes the parameter scale-free."""

    @staticmethod
    def _noise(values, actions=None):
        built = build_botorch_gp(
            actions=_grid() if actions is None else actions, values=values,
            noise_std=NOISE_STD, signal_var=1.0, length_scale=0.2
        )
        return built.likelihood.noise.detach().numpy()

    def test_the_stored_noise_is_the_squared_fraction(self, actions):
        # after Standardize the data has unit variance, so a noise standard
        # deviation of 5% of the spread is a variance of 0.05^2
        assert self._noise(_bump(actions, PEAK)) == pytest.approx(NOISE_STD ** 2)

    def test_rescaling_the_data_does_not_change_it(self, actions):
        """The load-bearing claim: 1000x larger measurements, same noise level."""
        values = _bump(actions, PEAK)
        assert self._noise(values) == pytest.approx(self._noise(1000.0 * values))

    def test_shifting_the_data_does_not_change_it(self, actions):
        # a shift moves the mean, not the spread
        values = _bump(actions, PEAK)
        assert self._noise(values) == pytest.approx(self._noise(values + 50.0))

    def test_the_posterior_rescales_with_the_data(self, actions):
        """End to end: the same model, read out in whichever units it was fed."""
        values = _bump(actions, PEAK)
        probe = torch.as_tensor(_grid(4), dtype=torch.float64)
        with torch.no_grad():
            small = build_botorch_gp(actions, values, NOISE_STD, 1.0, 0.2
                                     ).posterior(probe).mean.numpy()
            large = build_botorch_gp(actions, 1000.0 * values, NOISE_STD, 1.0, 0.2
                                     ).posterior(probe).mean.numpy()
        np.testing.assert_allclose(1000.0 * small, large, rtol=1e-8)

    def test_a_single_measurement_gives_a_finite_noise(self, actions):
        """One point has no sample standard deviation, so the spread falls back
        to 1 instead of propagating a NaN into the likelihood."""
        noise = self._noise(np.array([1.0]), actions=actions[:1])
        assert np.all(np.isfinite(noise))


# ---- BoTorchGP structure ---------------------------------------------------

class TestModelStructure:

    def test_it_wraps_one_single_task_gp(self, gp):
        """One objective, one GP -- no ModelListGP wrapper."""
        assert isinstance(gp.model, SingleTaskGP)

    def test_the_model_holds_the_objectives_measurements(self, gp, reward):
        assert gp.model.train_inputs[0].shape[0] == len(reward.ydata)

    def test_it_is_fit_on_the_standardized_values(self, gp, reward):
        """Not the raw ones: a minimized objective is already sign-flipped by
        the time a GP sees it."""
        np.testing.assert_allclose(_targets(gp.model), reward.standard_y)

    def test_it_is_fit_on_the_objectives_own_action_frame(self, gp, reward):
        np.testing.assert_allclose(gp.model.train_inputs[0].numpy(),
                                   reward.normalized_x)

    def test_the_model_is_in_eval_mode(self, gp):
        assert not gp.model.training

    def test_fitting_moves_the_lengthscales_off_their_prior(self, reward):
        def lengthscales(m):
            return m.model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()

        fixed  = BoTorchGP(reward, noise_std=NOISE_STD, fit_hyperparameters=False)
        fitted = BoTorchGP(reward, noise_std=NOISE_STD, fit_hyperparameters=True)

        assert np.allclose(lengthscales(fixed), BoTorchGP.LENGTH_SCALE)
        assert not np.allclose(lengthscales(fitted), BoTorchGP.LENGTH_SCALE)


# ---- posterior_at ----------------------------------------------------------

class TestPosteriorAt:

    def test_shapes_are_n_by_one(self, gp):
        """One objective, so the column axis is kept but is width 1 -- the same
        (n, m) contract as DecoupledMOGP."""
        mu, std = gp.posterior_at(np.array([[1.0, 0.0], [5.0, 1.5], [9.0, -1.0]]))
        assert mu.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_a_single_action_is_promoted_to_one_row(self, gp):
        mu, std = gp.posterior_at(np.array([5.0, 0.0]))
        assert mu.shape == (1, 1)
        assert std.shape == (1, 1)

    def test_standard_deviations_are_positive(self, gp):
        _, std = gp.posterior_at(_grid(5))
        assert np.all(std > 0)

    def test_evaluates_off_the_measured_actions(self, gp, actions):
        """Nothing snaps to a grid: midpoints between measurements are fine."""
        midpoints = 0.5 * (actions[:-1] + actions[1:])
        mu, _ = gp.posterior_at(midpoints)
        assert mu.shape == (len(actions) - 1, 1)
        assert np.all(np.isfinite(mu))

    def test_chunking_does_not_change_the_answer(self, gp, actions):
        whole = gp.posterior_at(actions, chunk=4096)
        split = gp.posterior_at(actions, chunk=7)
        np.testing.assert_allclose(whole[0], split[0])
        np.testing.assert_allclose(whole[1], split[1])

    def test_normalized_flag_selects_the_action_frame(self, gp, reward):
        """Passing raw actions equals passing them pre-normalized by hand."""
        raw = np.array([[1.0, 0.0], [5.0, 1.5]])
        from_raw  = gp.posterior_at(raw, normalized=False)
        from_norm = gp.posterior_at(reward.xtransform(raw), normalized=True)
        np.testing.assert_allclose(from_raw[0], from_norm[0])
        np.testing.assert_allclose(from_raw[1], from_norm[1])

    def test_raw_actions_are_not_silently_treated_as_normalized(self, gp):
        """The raw box is not [0, 1]^2, so the two frames must disagree."""
        raw = np.array([[7.5, 1.0]])
        assert not np.allclose(gp.posterior_at(raw, normalized=False)[0],
                               gp.posterior_at(raw, normalized=True)[0])

    def test_near_noiseless_posterior_passes_through_the_data(
            self, interpolating_gp, reward, actions):
        """With negligible noise the GP interpolates its measurements."""
        mu, _ = interpolating_gp.posterior_at(actions)
        np.testing.assert_allclose(mu[:, 0], reward.standard_y,
                                   atol=0.02 * reward.standard_y.std())

    def test_uncertainty_is_lower_at_measured_actions(self, gp, actions):
        """A measured corner is better known than a point outside the data."""
        _, at_data  = gp.posterior_at(actions[:1])
        _, off_data = gp.posterior_at(np.array([[20.0, 8.0]]))
        assert np.all(at_data[0] < off_data[0])

    def test_reports_maximization_space(self, cost_gp):
        """A minimized objective still reads larger-is-better here, matching
        Objective.standard_y."""
        mu, _ = cost_gp.posterior_at(np.vstack([VALLEY, PEAK]))
        assert mu[0, 0] > mu[1, 0]


class TestPosteriorAtRawUnits:
    """`raw=True` reports in the units the measurements were taken in, which is
    where a mean and a standard deviation stop transforming the same way."""

    def test_the_minimized_objective_reads_lower_is_better(self, cost_gp):
        mu, _ = cost_gp.posterior_at(np.vstack([VALLEY, PEAK]), raw=True)
        # back in its own units, so VALLEY is now the *smaller* value
        assert mu[0, 0] < mu[1, 0]

    def test_the_maximized_objective_is_unchanged_in_direction(self, gp):
        mu, _ = gp.posterior_at(np.vstack([PEAK, VALLEY]), raw=True)
        assert mu[0, 0] > mu[1, 0]

    def test_it_lands_in_the_range_of_the_measurements(self, gp, reward):
        mu, _ = gp.posterior_at(_grid(5), raw=True)
        assert mu.min() >= reward.ydata.min() - 0.5 * reward.ydata.std()
        assert mu.max() <= reward.ydata.max() + 0.5 * reward.ydata.std()

    def test_standard_deviations_stay_positive(self, cost_gp):
        """cost's ytransform carries a negative scale, so inv() would report a
        negative standard deviation here."""
        _, std = cost_gp.posterior_at(_grid(5), raw=True)
        assert np.all(std > 0)

    def test_a_standard_deviation_is_scaled_not_shifted(self, cost_gp, cost):
        """The whole point of inv_scale: inv() would add the mean back on."""
        _, std     = cost_gp.posterior_at(_grid(5))
        _, std_raw = cost_gp.posterior_at(_grid(5), raw=True)
        np.testing.assert_allclose(std_raw[:, 0], std[:, 0] * cost.ydata.std())

    def test_it_matches_converting_by_hand(self, cost_gp, cost):
        mu, std = cost_gp.posterior_at(_grid(4))
        by_hand = cost.to_raw(mu, std)
        got = cost_gp.posterior_at(_grid(4), raw=True)
        np.testing.assert_allclose(got[0], by_hand[0])
        np.testing.assert_allclose(got[1], by_hand[1])


# ---- best_actions ----------------------------------------------------------

class TestBestActions:

    def test_shapes(self, gp, actions):
        """One objective, so m = 1: a single row of actions and length-1
        moments, keeping the same contract as DecoupledMOGP."""
        best, mu, std = gp.best_actions(num_restarts=4, raw_samples=128)
        assert best.shape == (1, actions.shape[1])
        assert mu.shape == (1,)
        assert std.shape == (1,)

    def test_returns_raw_units_inside_the_action_box(self, gp):
        best, _, _ = gp.best_actions(num_restarts=4, raw_samples=128)
        assert np.all(best >= LOW - 1e-9)
        assert np.all(best <= HIGH + 1e-9)

    def test_recovers_the_maximized_objectives_peak(self, gp, reward):
        best, _, _ = gp.best_actions(num_restarts=8, raw_samples=256)
        # compared in the normalized frame, where one tolerance covers both
        # dimensions; GRID_SPACING is the distance between measured actions
        np.testing.assert_allclose(reward.xtransform(best)[0],
                                   reward.xtransform(PEAK)[0],
                                   atol=GRID_SPACING)

    def test_recovers_the_minimized_objectives_valley(self, cost_gp, cost):
        """A minimized objective is handled by the sign in ytransform, so the
        argmax of the posterior is the *lowest* raw cost."""
        best, _, _ = cost_gp.best_actions(num_restarts=8, raw_samples=256)
        np.testing.assert_allclose(cost.xtransform(best)[0],
                                   cost.xtransform(VALLEY)[0],
                                   atol=GRID_SPACING)

    def test_values_match_the_posterior_at_those_actions(self, gp):
        best, mu, std = gp.best_actions(num_restarts=4, raw_samples=128)
        mu_full, std_full = gp.posterior_at(best)
        np.testing.assert_allclose(mu, np.diag(mu_full))
        np.testing.assert_allclose(std, np.diag(std_full))

    def test_beats_every_measured_action(self, gp, actions):
        """Optimizing the continuous box can only match or beat the grid."""
        _, mu, _ = gp.best_actions(num_restarts=8, raw_samples=256)
        assert np.all(mu >= gp.posterior_at(actions)[0].max(axis=0) - 1e-6)

    def test_raw_units_leave_the_actions_alone(self, gp):
        """Only the values change; the actions are in raw units either way."""
        best, _, _     = gp.best_actions(num_restarts=4, raw_samples=128)
        best_raw, _, _ = gp.best_actions(num_restarts=4, raw_samples=128, raw=True)
        np.testing.assert_allclose(best, best_raw, atol=1e-6)

    def test_raw_values_match_the_posterior_at_those_actions(self, cost_gp):
        best, mu, std = cost_gp.best_actions(num_restarts=4, raw_samples=128, raw=True)
        mu_full, std_full = cost_gp.posterior_at(best, raw=True)
        np.testing.assert_allclose(mu, np.diag(mu_full))
        np.testing.assert_allclose(std, np.diag(std_full))

    def test_the_raw_optimum_is_the_smallest_predicted_cost(self, cost_gp):
        """Reported in raw units, a minimized objective's optimum is a minimum."""
        _, mu, _ = cost_gp.best_actions(num_restarts=8, raw_samples=256, raw=True)
        assert np.all(mu <= cost_gp.posterior_at(_grid(5), raw=True)[0].min() + 1e-6)


# ---- update_feedback -------------------------------------------------------

class TestUpdateFeedback:

    def test_refits_against_new_measurements(self, reward):
        model = BoTorchGP(reward, noise_std=1e-4, fit_hyperparameters=False)
        probe = np.array([[5.0, 0.0]])
        before = model.posterior_at(probe)[0].copy()

        # a measurement far from the current posterior, at the probe itself
        reward.add_points(probe, np.array([5.0]))
        model.update_feedback(reward)

        assert not np.allclose(before, model.posterior_at(probe)[0])

    def test_it_swaps_in_the_objective_it_was_given(self, gp, cost):
        gp.update_feedback(cost)
        assert gp.objective is cost
        np.testing.assert_allclose(_targets(gp.model), cost.standard_y)

    def test_returns_the_rebuilt_model_in_eval_mode(self, gp, reward):
        returned = gp.update_feedback(reward)
        assert returned is gp.model
        assert not returned.training


# ---- hyperparameters -------------------------------------------------------

KEYS = {'lengthscale', 'signal_var', 'noise_var', 'standardize_scale'}


class TestGpHyperparameters:
    """The four numbers that, with the training data, determine the posterior:
    the ARD lengthscales and outputscale of the ScaleKernel(RBFKernel), the
    fixed observation noise, and the scale Standardize divided the values by.
    ZeroMean contributes no parameter, and Standardize's offset is zero because
    `standard_y` is already centered, so nothing else is left to report."""

    @pytest.fixture
    def built(self, actions):
        return build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise_std=NOISE_STD, signal_var=3.0, length_scale=0.7
        )

    def test_it_reports_exactly_these_four(self, built):
        assert set(gp_hyperparameters(built)) == KEYS

    def test_they_are_the_ones_the_model_was_built_with(self, built):
        hypers = gp_hyperparameters(built)
        assert hypers['lengthscale'] == pytest.approx(0.7)
        assert hypers['signal_var'] == pytest.approx(3.0)

    def test_one_lengthscale_per_action_dimension(self, built, actions):
        assert gp_hyperparameters(built)['lengthscale'].shape == (actions.shape[1],)

    def test_nothing_torch_escapes(self, built):
        """The module's boundary contract: numpy out, never a tensor."""
        hypers = gp_hyperparameters(built)
        assert isinstance(hypers['lengthscale'], np.ndarray)
        assert isinstance(hypers['signal_var'], float)
        assert isinstance(hypers['noise_var'], float)
        assert isinstance(hypers['standardize_scale'], float)

    def test_it_reads_the_models_own_tensors(self, built):
        kernel = built.covar_module
        hypers = gp_hyperparameters(built)
        np.testing.assert_allclose(
            hypers['lengthscale'],
            kernel.base_kernel.lengthscale.detach().numpy().ravel()
        )
        assert hypers['signal_var'] == pytest.approx(kernel.outputscale.item())
        assert hypers['standardize_scale'] == \
            pytest.approx(built.outcome_transform.stdvs.item())

    def test_the_noise_is_the_squared_fraction(self, built):
        """Post-Standardize the data has unit variance, so a noise standard
        deviation of 5% of the spread reads as a variance of 0.05^2."""
        assert gp_hyperparameters(built)['noise_var'] == pytest.approx(NOISE_STD ** 2)

    def test_the_standardize_scale_is_the_sample_spread(self, built, actions):
        """Standardize uses the sample (ddof=1) standard deviation, which is
        what makes the reported variances post-Standardize quantities."""
        assert gp_hyperparameters(built)['standardize_scale'] == \
            pytest.approx(_bump(actions, PEAK).std(ddof=1))

    def test_the_scale_converts_the_noise_back_to_the_fed_units(self, built, actions):
        """The documented relation: multiplying by standardize_scale^2 undoes
        Standardize, recovering the train_Yvar build_botorch_gp pinned."""
        hypers = gp_hyperparameters(built)
        spread = _bump(actions, PEAK).std(ddof=1)
        assert hypers['noise_var'] * hypers['standardize_scale'] ** 2 == \
            pytest.approx((NOISE_STD * spread) ** 2)


class TestGetFittedHyperparameters:

    def test_it_does_not_reach_through_a_missing_attribute(self, gp):
        """BoTorchGP.model *is* the SingleTaskGP; there is no sub-model under
        it, so nothing here may go looking for one."""
        assert set(gp.get_fitted_hyperparameters()) == KEYS

    def test_it_matches_the_module_level_reader(self, gp):
        method = gp.get_fitted_hyperparameters()
        direct = gp_hyperparameters(gp.model)
        np.testing.assert_allclose(method['lengthscale'], direct['lengthscale'])
        assert method['signal_var'] == pytest.approx(direct['signal_var'])

    def test_an_unfitted_gp_reports_the_starting_values(self, reward):
        hypers = BoTorchGP(reward, noise_std=NOISE_STD, fit_hyperparameters=False
                           ).get_fitted_hyperparameters()
        assert hypers['lengthscale'] == pytest.approx(BoTorchGP.LENGTH_SCALE)
        assert hypers['signal_var'] == pytest.approx(BoTorchGP.SIGNAL_VAR)

    def test_a_fitted_gp_reports_values_it_moved_to(self, gp):
        hypers = gp.get_fitted_hyperparameters()
        assert not np.allclose(hypers['lengthscale'], BoTorchGP.LENGTH_SCALE)
        assert np.all(hypers['lengthscale'] > 0)
        assert hypers['signal_var'] > 0

    def test_a_one_dimensional_objective_gets_one_lengthscale(self, actions):
        one_d = Objective.from_data(
            actions=actions[:, :1], values=_bump(actions, PEAK),
            maximize=True, name='reward'
        )
        gp = BoTorchGP(one_d, noise_std=NOISE_STD, fit_hyperparameters=True)
        assert gp.get_fitted_hyperparameters()['lengthscale'].shape == (1,)

    def test_the_objectives_units_do_not_change_them(self, actions):
        """Objective standardizes before a GP ever sees the values, so the same
        measurements in different units must fit the same hyperparameters."""
        def fitted(scale):
            obj = Objective.from_data(
                actions=actions, values=scale * _bump(actions, PEAK),
                maximize=True, name='reward'
            )
            return BoTorchGP(obj, noise_std=NOISE_STD, fit_hyperparameters=True
                             ).get_fitted_hyperparameters()

        small, large = fitted(1.0), fitted(1000.0)
        np.testing.assert_allclose(small['lengthscale'], large['lengthscale'])
        assert small['signal_var'] == pytest.approx(large['signal_var'])
        assert small['noise_var'] == pytest.approx(large['noise_var'])

    def test_it_tracks_a_refit_after_new_measurements(self, reward):
        model = BoTorchGP(reward, noise_std=NOISE_STD, fit_hyperparameters=True)
        before = model.get_fitted_hyperparameters()['lengthscale'].copy()

        # a measurement contradicting the bump, so the fit has to move
        reward.add_points(np.array([[5.0, 0.0]]), np.array([5.0]))
        model.update_feedback(reward)

        assert not np.allclose(before, model.get_fitted_hyperparameters()['lengthscale'])
