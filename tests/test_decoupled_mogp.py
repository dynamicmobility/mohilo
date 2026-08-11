"""Tests for DecoupledMOGP.

The class makes four promises worth pinning down: one independent GP per
objective, a posterior evaluable at *arbitrary* actions (not just measured
ones), a shared normalized action frame that raw actions are mapped through,
and a maximization space that hides the sign of a minimized objective. Each
test below states one of those.
"""

import numpy as np
import pytest

from pypolar.optimization.gp import DecoupledMOGP
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
    return DecoupledMOGP(objectives, fit_hyperparameters=False, noise_std=1e-4)


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

        assert np.allclose(lengthscales(fixed), DecoupledMOGP.LENGTH_SCALE)
        assert not np.allclose(lengthscales(fitted), DecoupledMOGP.LENGTH_SCALE)


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


# ---- update_feedback -------------------------------------------------------

class TestUpdateFeedback:

    def test_refits_against_new_measurements(self, objectives, actions):
        model = DecoupledMOGP(objectives, fit_hyperparameters=False, noise_std=1e-4)
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
