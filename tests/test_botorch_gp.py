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

from dataclasses import fields

import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ZeroMean

from pypolar.optimization.gp import (
    LENGTH_SCALE,
    SIGNAL_VAR,
    BoTorchGP,
    NoiseModel,
    build_botorch_gp,
    gp_hyperparameters,
)
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


def _noisy(values, rel, seed=0):
    """`values` plus Gaussian noise of `rel` times their own spread."""
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, rel * values.std(), size=values.shape)


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
    return BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=True)


@pytest.fixture
def cost_gp(cost):
    return BoTorchGP(cost, noise=NOISE_STD, fit_hyperparameters=True)


@pytest.fixture
def interpolating_gp(reward):
    """Near-noiseless, so the posterior must pass through the measurements."""
    return BoTorchGP(reward, noise=1e-4, fit_hyperparameters=False)


# ---- build_botorch_gp ------------------------------------------------------

class TestBuildBotorchGpStructure:
    """The kernel, mean and likelihood the rest of the package assumes."""

    @pytest.fixture
    def built(self, actions):
        return build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise=NOISE_STD, signal_var=1.0, length_scale=0.2
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
            noise=NOISE_STD, signal_var=1.0, length_scale=0.2
        )
        assert built.covar_module.base_kernel.lengthscale.shape[-1] == 1

    def test_the_starting_hyperparameters_are_the_requested_ones(self, actions):
        built = build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise=NOISE_STD, signal_var=3.0, length_scale=0.7
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
            noise=NOISE_STD, signal_var=1.0, length_scale=0.2
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


class TestBuildBotorchGpFittedNoise:
    """noise=None leaves the noise for the fit to determine instead of
    pinning it, which is a different likelihood rather than a different value."""

    @pytest.fixture
    def built(self, actions):
        return build_botorch_gp(
            actions=actions, values=_bump(actions, PEAK),
            noise=None, signal_var=1.0, length_scale=0.2
        )

    def test_the_likelihood_is_learnable(self, built):
        """Not FixedNoiseGaussianLikelihood, which stores noise as a buffer and
        leaves nothing for fit_gpytorch_mll to move."""
        assert isinstance(built.likelihood, GaussianLikelihood)
        assert not isinstance(built.likelihood, FixedNoiseGaussianLikelihood)

    def test_the_noise_joins_the_fitted_hyperparameters(self, built, actions):
        pinned = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.2)
        assert 'likelihood.noise_covar.raw_noise' in dict(built.named_hyperparameters())
        assert 'likelihood.noise_covar.raw_noise' not in dict(pinned.named_hyperparameters())

    def test_it_carries_no_prior(self, built):
        """The likelihood is passed explicitly for this reason: BoTorch's own
        default attaches a LogNormal noise prior centered low, which would
        decide the noise instead of the data."""
        assert [p[0] for p in built.named_priors()] == []

    def test_the_hyperparameter_fields_are_unchanged(self, built):
        assert _field_names(gp_hyperparameters(built)) == FIELDS

    def test_everything_is_still_float64(self, built):
        assert built.likelihood.noise.dtype == torch.float64


class TestBuildBotorchGpLengthscaleBound:
    """min_length_scale floors every ARD lengthscale, in the normalized frame."""

    def test_no_bound_is_the_default(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.2)
        assert built.covar_module.base_kernel.lengthscale.detach().numpy() == \
            pytest.approx(0.2)

    def test_the_bound_floors_the_lengthscale(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.2,
                                 min_length_scale=0.3)
        assert np.all(built.covar_module.base_kernel.lengthscale.detach().numpy() >= 0.3)

    def test_a_start_below_the_bound_is_moved_inside_it(self, actions):
        """A GreaterThan constraint cannot represent its own edge, so a start at
        or below the bound would raise rather than clamp."""
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.2,
                                 min_length_scale=0.3)
        assert built.covar_module.base_kernel.lengthscale.detach().numpy() == \
            pytest.approx(0.45)

    def test_a_start_above_the_bound_is_kept(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.7,
                                 min_length_scale=0.3)
        assert built.covar_module.base_kernel.lengthscale.detach().numpy() == \
            pytest.approx(0.7)

    def test_a_per_dimension_start_is_floored_elementwise(self, actions):
        """Freezing a fitted kernel hands back one lengthscale per action
        dimension, so the start is a vector rather than a scalar."""
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0,
                                 np.array([0.2, 0.9]), min_length_scale=0.3)
        np.testing.assert_allclose(
            built.covar_module.base_kernel.lengthscale.detach().numpy().ravel(),
            [0.45, 0.9]
        )

    def test_it_stays_float64(self, actions):
        """The constraint stores its bound in single precision, so the
        transformed lengthscale has to promote back."""
        built = build_botorch_gp(actions, _bump(actions, PEAK), NOISE_STD, 1.0, 0.2,
                                 min_length_scale=0.3)
        assert built.covar_module.base_kernel.lengthscale.dtype == torch.float64


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

        fixed  = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=False)
        fitted = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=True)

        assert np.allclose(lengthscales(fixed), LENGTH_SCALE)
        assert not np.allclose(lengthscales(fitted), LENGTH_SCALE)


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


# ---- sample_paths ----------------------------------------------------------

class TestSamplePaths:
    """Draws from the *joint* posterior, where `posterior_at` reports only the
    marginals. The two must agree point by point on mean and spread, and differ
    exactly in that the draws carry the covariance between actions.

    Every test seeds torch first, since the draws come from its global generator.
    """

    def test_shape_is_q_by_n_by_one(self, gp):
        """One objective, so the column axis is kept but is width 1 -- the same
        (n, m) column contract `posterior_at` keeps."""
        assert gp.sample_paths(_grid(3), num_paths=5).shape == (5, 9, 1)

    def test_a_single_action_is_promoted_to_one_row(self, gp):
        assert gp.sample_paths(np.array([5.0, 0.0]), num_paths=4).shape == (4, 1, 1)

    def test_marginals_match_posterior_at(self, gp):
        """Averaged over enough draws, the paths reproduce the mean and standard
        deviation `posterior_at` reports."""
        X = _grid(3)
        torch.manual_seed(0)
        paths   = gp.sample_paths(X, num_paths=4000)
        mu, std = gp.posterior_at(X)
        np.testing.assert_allclose(paths.mean(axis=0), mu, atol=0.05)
        np.testing.assert_allclose(paths.std(axis=0), std, rtol=0.1)

    def test_draws_are_joint_rather_than_independent(self, gp):
        """Two actions a hair apart are effectively the same point of the
        function, so their values must move together across draws. Sampling each
        from its own marginal would leave them uncorrelated."""
        X = np.array([[5.0, 0.0], [5.0 + 1e-3, 0.0]])
        torch.manual_seed(0)
        paths = gp.sample_paths(X, num_paths=200)[:, :, 0]
        assert np.corrcoef(paths[:, 0], paths[:, 1])[0, 1] > 0.99

    def test_correlation_falls_off_with_distance(self, gp):
        """The covariance being sampled is the kernel's, so a far pair is less
        correlated than a near one."""
        near = np.array([[5.0, 0.0], [5.2, 0.0]])
        far  = np.array([[5.0, 0.0], [0.0, -2.0]])
        torch.manual_seed(0)
        near_paths = gp.sample_paths(near, num_paths=400)[:, :, 0]
        torch.manual_seed(0)
        far_paths  = gp.sample_paths(far, num_paths=400)[:, :, 0]
        assert (np.corrcoef(*near_paths.T)[0, 1] > np.corrcoef(*far_paths.T)[0, 1])

    def test_normalized_flag_selects_the_action_frame(self, gp, reward):
        """Passing raw actions equals passing them pre-normalized by hand."""
        raw = np.array([[1.0, 0.0], [5.0, 1.5]])
        torch.manual_seed(0)
        from_raw = gp.sample_paths(raw, num_paths=3)
        torch.manual_seed(0)
        from_norm = gp.sample_paths(reward.xtransform(raw), num_paths=3, normalized=True)
        np.testing.assert_allclose(from_raw, from_norm)

    def test_raw_actions_are_not_silently_treated_as_normalized(self, gp):
        """The raw box is not [0, 1]^2, so the two frames must disagree."""
        raw = np.array([[7.5, 1.0]])
        torch.manual_seed(0)
        from_raw = gp.sample_paths(raw, num_paths=200)
        torch.manual_seed(0)
        as_norm = gp.sample_paths(raw, num_paths=200, normalized=True)
        assert not np.allclose(from_raw.mean(axis=0), as_norm.mean(axis=0))

    def test_raw_units_match_converting_by_hand(self, cost_gp, cost):
        torch.manual_seed(0)
        standard = cost_gp.sample_paths(_grid(3), num_paths=3)
        torch.manual_seed(0)
        got = cost_gp.sample_paths(_grid(3), num_paths=3, raw=True)
        np.testing.assert_allclose(got, cost.to_raw(standard))

    def test_raw_restores_the_sign_of_a_minimized_objective(self, cost_gp):
        """Maximization space reads larger-is-better at VALLEY; its own units
        read lower-is-better there."""
        X = np.vstack([VALLEY, PEAK])
        torch.manual_seed(0)
        standard = cost_gp.sample_paths(X, num_paths=400)[:, :, 0].mean(axis=0)
        torch.manual_seed(0)
        raw = cost_gp.sample_paths(X, num_paths=400, raw=True)[:, :, 0].mean(axis=0)
        assert standard[0] > standard[1]
        assert raw[0] < raw[1]

    def test_a_near_noiseless_gp_pins_its_paths_to_the_data(
            self, interpolating_gp, reward, actions):
        """With negligible noise every draw passes through the measurements, so
        the paths have almost no spread there."""
        torch.manual_seed(0)
        paths = interpolating_gp.sample_paths(actions, num_paths=50)[:, :, 0]
        assert paths.std(axis=0).max() < 0.02 * reward.standard_y.std()


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
        model = BoTorchGP(reward, noise=1e-4, fit_hyperparameters=False)
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


class TestStateDict:
    """A state dict is the fit, so a model restored from one is not fitted
    again: the same measurements plus the same tensors give back the same
    posterior, and a fitted noise no longer needs a fit to determine it."""

    def test_restores_the_posterior_it_was_fitted_to(self, gp, reward, actions):
        restored = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=False,
                             state_dict=gp.model.state_dict())
        mu, std = gp.posterior_at(actions)
        mu_restored, std_restored = restored.posterior_at(actions)

        np.testing.assert_allclose(mu_restored, mu)
        np.testing.assert_allclose(std_restored, std)

    def test_a_fitted_noise_needs_no_fit_when_restored(self, reward):
        """The marginal likelihood determines a fitted noise, so a fit is
        normally required; a state dict carries the value it landed on."""
        fitted = BoTorchGP(reward, noise=NoiseModel.prior(0.3))

        restored = BoTorchGP(reward, noise=NoiseModel.prior(0.3),
                             fit_hyperparameters=False,
                             state_dict=fitted.model.state_dict())

        assert (restored.get_fitted_hyperparameters().noise_var
                == fitted.get_fitted_hyperparameters().noise_var)

    def test_without_one_a_fitted_noise_still_refuses_not_to_fit(self, reward):
        with pytest.raises(ValueError):
            BoTorchGP(reward, noise=NoiseModel.fitted(), fit_hyperparameters=False)


# ---- hyperparameters -------------------------------------------------------

FIELDS = {'lengthscale', 'signal_var', 'noise_var', 'standardize_scale'}


def _field_names(hypers):
    return {f.name for f in fields(hypers)}


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
            noise=NOISE_STD, signal_var=3.0, length_scale=0.7
        )

    def test_it_reports_exactly_these_four(self, built):
        assert _field_names(gp_hyperparameters(built)) == FIELDS

    def test_they_are_the_ones_the_model_was_built_with(self, built):
        hypers = gp_hyperparameters(built)
        assert hypers.lengthscale == pytest.approx(0.7)
        assert hypers.signal_var == pytest.approx(3.0)

    def test_one_lengthscale_per_action_dimension(self, built, actions):
        assert gp_hyperparameters(built).lengthscale.shape == (actions.shape[1],)

    def test_nothing_torch_escapes(self, built):
        """The module's boundary contract: numpy out, never a tensor."""
        hypers = gp_hyperparameters(built)
        assert isinstance(hypers.lengthscale, np.ndarray)
        assert isinstance(hypers.signal_var, float)
        assert isinstance(hypers.noise_var, float)
        assert isinstance(hypers.standardize_scale, float)

    def test_it_reads_the_models_own_tensors(self, built):
        kernel = built.covar_module
        hypers = gp_hyperparameters(built)
        np.testing.assert_allclose(
            hypers.lengthscale,
            kernel.base_kernel.lengthscale.detach().numpy().ravel()
        )
        assert hypers.signal_var == pytest.approx(kernel.outputscale.item())
        assert hypers.standardize_scale == \
            pytest.approx(built.outcome_transform.stdvs.item())

    def test_the_noise_is_the_squared_fraction(self, built):
        """Post-Standardize the data has unit variance, so a noise standard
        deviation of 5% of the spread reads as a variance of 0.05^2."""
        assert gp_hyperparameters(built).noise_var == pytest.approx(NOISE_STD ** 2)

    def test_the_standardize_scale_is_the_sample_spread(self, built, actions):
        """Standardize uses the sample (ddof=1) standard deviation, which is
        what makes the reported variances post-Standardize quantities."""
        assert gp_hyperparameters(built).standardize_scale == \
            pytest.approx(_bump(actions, PEAK).std(ddof=1))

    def test_the_scale_converts_the_noise_back_to_the_fed_units(self, built, actions):
        """The documented relation: multiplying by standardize_scale^2 undoes
        Standardize, recovering the train_Yvar build_botorch_gp pinned."""
        hypers = gp_hyperparameters(built)
        spread = _bump(actions, PEAK).std(ddof=1)
        assert hypers.noise_var * hypers.standardize_scale ** 2 == \
            pytest.approx((NOISE_STD * spread) ** 2)


class TestGetFittedHyperparameters:

    def test_it_does_not_reach_through_a_missing_attribute(self, gp):
        """BoTorchGP.model *is* the SingleTaskGP; there is no sub-model under
        it, so nothing here may go looking for one."""
        assert _field_names(gp.get_fitted_hyperparameters()) == FIELDS

    def test_it_matches_the_module_level_reader(self, gp):
        method = gp.get_fitted_hyperparameters()
        direct = gp_hyperparameters(gp.model)
        np.testing.assert_allclose(method.lengthscale, direct.lengthscale)
        assert method.signal_var == pytest.approx(direct.signal_var)

    def test_an_unfitted_gp_reports_the_starting_values(self, reward):
        hypers = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=False
                           ).get_fitted_hyperparameters()
        assert hypers.lengthscale == pytest.approx(LENGTH_SCALE)
        assert hypers.signal_var == pytest.approx(SIGNAL_VAR)

    def test_a_fitted_gp_reports_values_it_moved_to(self, gp):
        hypers = gp.get_fitted_hyperparameters()
        assert not np.allclose(hypers.lengthscale, LENGTH_SCALE)
        assert np.all(hypers.lengthscale > 0)
        assert hypers.signal_var > 0

    def test_a_one_dimensional_objective_gets_one_lengthscale(self, actions):
        one_d = Objective.from_data(
            actions=actions[:, :1], values=_bump(actions, PEAK),
            maximize=True, name='reward'
        )
        gp = BoTorchGP(one_d, noise=NOISE_STD, fit_hyperparameters=True)
        assert gp.get_fitted_hyperparameters().lengthscale.shape == (1,)

    def test_the_objectives_units_do_not_change_them(self, actions):
        """Objective standardizes before a GP ever sees the values, so the same
        measurements in different units must fit the same hyperparameters."""
        def fitted(scale):
            obj = Objective.from_data(
                actions=actions, values=scale * _bump(actions, PEAK),
                maximize=True, name='reward'
            )
            return BoTorchGP(obj, noise=NOISE_STD, fit_hyperparameters=True
                             ).get_fitted_hyperparameters()

        small, large = fitted(1.0), fitted(1000.0)
        np.testing.assert_allclose(small.lengthscale, large.lengthscale)
        assert small.signal_var == pytest.approx(large.signal_var)
        assert small.noise_var == pytest.approx(large.noise_var)

    def test_it_tracks_a_refit_after_new_measurements(self, reward):
        model = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=True)
        before = model.get_fitted_hyperparameters().lengthscale.copy()

        # a measurement contradicting the bump, so the fit has to move
        reward.add_points(np.array([[5.0, 0.0]]), np.array([5.0]))
        model.update_feedback(reward)

        assert not np.allclose(before, model.get_fitted_hyperparameters().lengthscale)


# ---- fitting the noise -----------------------------------------------------

class TestFittedNoise:
    """With noise=None the noise is fitted by marginal likelihood, so it has
    to track how noisy the measurements actually are."""

    @staticmethod
    def _fitted(values, actions, maximize=True):
        obj = Objective.from_data(actions=actions, values=values,
                                  maximize=maximize, name='reward')
        gp = BoTorchGP(obj, noise=None, fit_hyperparameters=True)
        return gp.get_fitted_hyperparameters()

    def test_noisier_measurements_fit_a_larger_noise(self, actions):
        clean = _bump(actions, PEAK)
        fitted = [self._fitted(_noisy(clean, rel), actions).noise_var
                  for rel in (0.1, 0.3, 0.5)]
        assert fitted == sorted(fitted)

    def test_it_recovers_roughly_the_noise_that_was_added(self, actions):
        """Post-Standardize the values have unit variance, so the fitted
        noise_var is the squared fraction of the spread. Adding 30% of the clean
        spread makes it 30% / sqrt(1 + 0.3^2) of the noisy one."""
        noise_std = np.sqrt(self._fitted(_noisy(_bump(actions, PEAK), 0.3),
                                         actions).noise_var)
        assert noise_std == pytest.approx(0.3 / np.sqrt(1.09), rel=0.4)

    def test_noiseless_measurements_fit_the_smallest_allowed_noise(self, actions):
        """Nothing to explain, so the fit drives the noise to the likelihood's
        own floor rather than to zero, which would be singular."""
        assert self._fitted(_bump(actions, PEAK), actions).noise_var == \
            pytest.approx(1e-4)

    def test_the_objectives_units_do_not_change_it(self, actions):
        """The same scale-free promise the pinned path makes: noise_std is a
        fraction of the objective's own spread either way."""
        values = _noisy(_bump(actions, PEAK), 0.3)
        assert self._fitted(values, actions).noise_var == \
            pytest.approx(self._fitted(1000.0 * values, actions).noise_var)

    def test_a_minimized_objective_fits_the_same_noise(self, actions):
        """Objective flips the sign before a GP sees it, and a sign flip cannot
        change a spread."""
        values = _noisy(_bump(actions, PEAK), 0.3)
        assert self._fitted(values, actions, maximize=True).noise_var == \
            pytest.approx(self._fitted(-values, actions, maximize=False).noise_var)

    def test_it_refuses_to_leave_the_noise_undetermined(self, reward):
        """Without a fit there is nothing to set the noise, so the GP would
        silently keep the likelihood's arbitrary starting value."""
        with pytest.raises(ValueError, match='fit_hyperparameters'):
            BoTorchGP(reward, noise=None, fit_hyperparameters=False)

    def test_a_pinned_noise_still_reports_what_it_was_given(self, reward):
        """The default path is untouched: the fit never moves a pinned noise."""
        gp = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=True)
        assert gp.get_fitted_hyperparameters().noise_var == \
            pytest.approx(NOISE_STD ** 2)


class TestMinLengthScale:
    """min_length_scale floors the fitted lengthscales, in the normalized frame.
    A lengthscale below the design spacing is not identifiable, so the bound is
    what stops the fit from wandering there."""

    @staticmethod
    def _bounded(objective, noise=NOISE_STD):
        return BoTorchGP(objective, noise=noise, fit_hyperparameters=True,
                         min_length_scale=0.5)

    def test_the_default_is_no_bound(self, gp):
        assert gp.min_length_scale is None

    def test_the_fit_respects_it(self, reward):
        gp = self._bounded(reward)
        assert np.all(gp.get_fitted_hyperparameters().lengthscale >= 0.5)

    def test_it_is_a_floor_and_not_a_value(self, reward):
        """It binds on the dimension the free fit puts below it, and leaves the
        rest free to sit above it. Not *unchanged*, though: the ARD lengthscales
        share one marginal likelihood, so lifting one moves the others."""
        free  = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=True)
        free_ls  = free.get_fitted_hyperparameters().lengthscale
        bound_ls = self._bounded(reward).get_fitted_hyperparameters().lengthscale

        assert np.any(free_ls < 0.5)      # the bound has something to bind on here
        assert np.all(bound_ls >= 0.5)
        assert np.any(bound_ls > 0.5)     # and does not collapse the rest onto it

    def test_it_combines_with_a_fitted_noise(self, reward):
        hypers = self._bounded(reward, noise=None).get_fitted_hyperparameters()
        assert np.all(hypers.lengthscale >= 0.5)
        assert hypers.noise_var > 0

    def test_it_survives_a_refit(self, reward):
        gp = self._bounded(reward)
        reward.add_points(np.array([[5.0, 0.0]]), np.array([5.0]))
        gp.update_feedback(reward)
        assert np.all(gp.get_fitted_hyperparameters().lengthscale >= 0.5)

    def test_the_starting_hyperparameters_are_arguments_too(self, reward):
        """length_scale and signal_var are per-instance, so two GPs on the same
        objective can start from different places."""
        gp = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=False,
                       length_scale=0.7, signal_var=3.0)
        hypers = gp.get_fitted_hyperparameters()
        assert hypers.lengthscale == pytest.approx(0.7)
        assert hypers.signal_var == pytest.approx(3.0)

    def test_the_defaults_are_the_module_constants(self, reward):
        gp = BoTorchGP(reward, noise=NOISE_STD, fit_hyperparameters=False)
        hypers = gp.get_fitted_hyperparameters()
        assert hypers.lengthscale == pytest.approx(LENGTH_SCALE)
        assert hypers.signal_var == pytest.approx(SIGNAL_VAR)


# ---- NoiseModel ------------------------------------------------------------

class TestNoiseModel:
    """One argument carrying the three noise representations, so a pinned noise
    and a prior on a fitted one cannot be requested at the same time."""

    def test_pinned_selects_a_fixed_likelihood(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK),
                                 NoiseModel.pinned(0.05), 1.0, 0.2)
        assert isinstance(built.likelihood, FixedNoiseGaussianLikelihood)
        assert built.likelihood.noise.mean().item() == pytest.approx(0.05 ** 2)

    def test_fitted_carries_no_prior(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK),
                                 NoiseModel.fitted(), 1.0, 0.2)
        assert isinstance(built.likelihood, GaussianLikelihood)
        assert [p[0] for p in built.named_priors()] == []

    def test_prior_attaches_a_lognormal(self, actions):
        built = build_botorch_gp(actions, _bump(actions, PEAK),
                                 NoiseModel.prior(0.3), 1.0, 0.2)
        assert 'likelihood.noise_covar.noise_prior' in \
            [p[0] for p in built.named_priors()]

    def test_the_prior_is_centered_on_the_median_noise_std(self, actions):
        """The prior is on the variance, so a median noise standard deviation of
        m has to become a median variance of m^2."""
        built = build_botorch_gp(actions, _bump(actions, PEAK),
                                 NoiseModel.prior(0.3), 1.0, 0.2)
        prior = built.likelihood.noise_covar.noise_prior
        assert np.exp(prior.loc.item()) == pytest.approx(0.3 ** 2)

    def test_the_prior_pulls_a_collapsing_fit_back(self, actions):
        """Noiseless data drives an unpriored fit onto the likelihood floor. A
        prior centered well above it must land higher than that."""
        clean = _bump(actions, PEAK)
        obj = Objective.from_data(actions=actions, values=clean,
                                  maximize=True, name='reward')
        free    = BoTorchGP(obj, noise=NoiseModel.fitted())
        priored = BoTorchGP(obj, noise=NoiseModel.prior(0.3))

        assert priored.get_fitted_hyperparameters().noise_var > \
            free.get_fitted_hyperparameters().noise_var

    def test_a_wider_prior_constrains_less(self, actions):
        """sigma is the prior width, so widening it lets the data pull further
        from the median."""
        obj = Objective.from_data(actions=actions, values=_bump(actions, PEAK),
                                  maximize=True, name='reward')
        tight = BoTorchGP(obj, noise=NoiseModel.prior(0.3, sigma=0.2))
        loose = BoTorchGP(obj, noise=NoiseModel.prior(0.3, sigma=3.0))

        assert tight.get_fitted_hyperparameters().noise_var > \
            loose.get_fitted_hyperparameters().noise_var

    def test_a_pinned_noise_cannot_also_carry_a_prior(self):
        with pytest.raises(ValueError, match='pinned'):
            NoiseModel(std=0.05, median=0.3)

    def test_a_prior_median_must_be_positive(self):
        """A LogNormal has positive support, so a non-positive median has no
        meaning and log() would return -inf rather than raise."""
        with pytest.raises(ValueError, match='positive'):
            NoiseModel.prior(0.0)

    def test_a_pinned_noise_cannot_be_negative(self):
        with pytest.raises(ValueError, match='non-negative'):
            NoiseModel.pinned(-0.1)

    def test_it_coerces_the_shorthands(self):
        assert NoiseModel.coerce(0.05) == NoiseModel.pinned(0.05)
        assert NoiseModel.coerce(None) == NoiseModel.fitted()
        model = NoiseModel.prior(0.3)
        assert NoiseModel.coerce(model) is model

    def test_the_wrapper_coerces_too(self, reward):
        """A bare float still means a pinned noise, so no call site had to change."""
        assert BoTorchGP(reward, noise=0.05).noise == NoiseModel.pinned(0.05)

    def test_is_fitted_splits_the_two_kinds(self):
        assert not NoiseModel.pinned(0.05).is_fitted
        assert NoiseModel.fitted().is_fitted
        assert NoiseModel.prior(0.3).is_fitted

    def test_it_is_immutable(self):
        """Frozen, so a NoiseModel shared between GPs cannot be edited under one
        of them."""
        with pytest.raises(Exception):
            NoiseModel.pinned(0.05).std = 0.2
