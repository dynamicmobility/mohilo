"""Tests for ScalarizedGP.

The class makes one claim: `w^T f` is itself a GP, read off an already-fitted
`DecoupledMOGP` with no refit. Its posterior therefore has mean `w^T mu` and
variance `sum_j w_j^2 sigma_j^2` -- the cross terms vanish because the
objectives are decoupled. The tests below pin that identity, the two limits it
implies (a one-hot `w` reproduces one objective's own GP, and the whole thing
shares the underlying models), and the deliberate absence of a raw frame.
"""

from dataclasses import fields

import numpy as np
import pytest
import torch

from pypolar.optimization.gp import DecoupledMOGP, ScalarizedGP
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

# Raw action box, deliberately neither unit nor centered, so a test that
# forgets the normalizing xtransform fails visibly.
LOW  = np.array([0.0, -2.0])
HIGH = np.array([10.0, 2.0])

PEAK   = np.array([7.5, 1.0])    # argmax of the maximized objective, raw units
VALLEY = np.array([2.5, -1.0])   # argmin of the minimized objective, raw units

WEIGHTS = np.array([0.6, 0.4])   # a generic interior point of the simplex


def _grid(n=8):
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
    """One maximized and one minimized objective, each with a known optimum."""
    objs = DecoupledObjectives.from_empty()
    objs.add_objective(Objective.from_data(
        actions=actions, values=_bump(actions, PEAK), maximize=True,
        name='reward', action_bounds=(LOW, HIGH)
    ))
    objs.add_objective(Objective.from_data(
        actions=actions, values=-_bump(actions, VALLEY), maximize=False,
        name='cost', action_bounds=(LOW, HIGH)
    ))
    return objs


@pytest.fixture
def mogp(objectives):
    return DecoupledMOGP(objectives, fit_hyperparameters=True)


@pytest.fixture
def scalar_gp(mogp):
    return mogp.scalarized(WEIGHTS)


@pytest.fixture
def test_actions():
    """Off-grid actions, so nothing passes by sitting on a measurement."""
    rng = np.random.default_rng(0)
    return rng.uniform(LOW, HIGH, size=(16, 2))


# ---- the scalarization is a read-out, not a fit ----------------------------

class TestConstruction:

    def test_shares_the_underlying_models(self, mogp, scalar_gp):
        assert scalar_gp.model is mogp.model

    def test_shares_the_objectives(self, mogp, scalar_gp):
        assert scalar_gp.objectives is mogp.objectives

    def test_hyperparameters_are_the_mogps_own(self, mogp, scalar_gp):
        # GPHyperparameters compares by identity, so check the fields
        for got, want in zip(scalar_gp.get_fitted_hyperparameters(),
                             mogp.get_fitted_hyperparameters()):
            for field in fields(got):
                assert np.array_equal(getattr(got, field.name),
                                      getattr(want, field.name))

    def test_rejects_the_wrong_number_of_weights(self, mogp):
        with pytest.raises(ValueError):
            mogp.scalarized([1.0, 0.0, 0.0])

    def test_constructor_and_method_agree(self, mogp):
        assert np.array_equal(ScalarizedGP(mogp, WEIGHTS).weights,
                              mogp.scalarized(WEIGHTS).weights)


# ---- mean w^T mu, variance sum_j w_j^2 sigma_j^2 ---------------------------

class TestPosteriorAt:

    def test_shape_is_one_column(self, scalar_gp, test_actions):
        mu, std = scalar_gp.posterior_at(test_actions)
        assert mu.shape == (len(test_actions), 1)
        assert std.shape == (len(test_actions), 1)

    def test_mean_is_the_weighted_sum_of_the_means(self, mogp, scalar_gp, test_actions):
        mu_mo, _ = mogp.posterior_at(test_actions)
        mu_w, _  = scalar_gp.posterior_at(test_actions)
        assert np.allclose(mu_w[:, 0], mu_mo @ WEIGHTS)

    def test_variance_uses_the_squared_weights(self, mogp, scalar_gp, test_actions):
        # diagonal covariance, so the cross terms are absent by construction
        _, std_mo = mogp.posterior_at(test_actions)
        _, std_w  = scalar_gp.posterior_at(test_actions)
        assert np.allclose(std_w[:, 0] ** 2, (std_mo ** 2) @ WEIGHTS ** 2)

    def test_a_one_hot_weight_reproduces_that_objectives_gp(self, mogp, test_actions):
        mu_mo, std_mo = mogp.posterior_at(test_actions)
        for j in range(len(mogp.objectives)):
            mu_w, std_w = mogp.scalarized(np.eye(2)[j]).posterior_at(test_actions)
            assert np.allclose(mu_w[:, 0], mu_mo[:, j])
            assert np.allclose(std_w[:, 0], std_mo[:, j])

    def test_normalized_and_raw_actions_agree(self, scalar_gp, test_actions, objectives):
        raw = scalar_gp.posterior_at(test_actions)
        nrm = scalar_gp.posterior_at(objectives.xtransform(test_actions), normalized=True)
        assert np.allclose(raw[0], nrm[0]) and np.allclose(raw[1], nrm[1])

    def test_chunking_does_not_change_the_answer(self, scalar_gp, test_actions):
        whole = scalar_gp.posterior_at(test_actions)
        split = scalar_gp.posterior_at(test_actions, chunk=3)
        assert np.allclose(whole[0], split[0]) and np.allclose(whole[1], split[1])

    def test_there_is_no_raw_frame(self, scalar_gp, test_actions):
        # w^T y mixes units, so no objective's ytransform can invert it
        with pytest.raises(TypeError):
            scalar_gp.posterior_at(test_actions, raw=True)


class TestSamplePaths:

    def test_shape(self, scalar_gp, test_actions):
        paths = scalar_gp.sample_paths(test_actions, num_paths=5)
        assert paths.shape == (5, len(test_actions), 1)

    def test_marginals_match_the_posterior(self, scalar_gp, test_actions):
        torch.manual_seed(0)
        paths = scalar_gp.sample_paths(test_actions, num_paths=4096)[..., 0]
        mu, std = scalar_gp.posterior_at(test_actions)
        assert np.allclose(paths.mean(axis=0), mu[:, 0], atol=0.05)
        assert np.allclose(paths.std(axis=0),  std[:, 0], atol=0.05)

    def test_there_is_no_raw_frame(self, scalar_gp, test_actions):
        with pytest.raises(TypeError):
            scalar_gp.sample_paths(test_actions, num_paths=2, raw=True)


# ---- one optimization rather than m ----------------------------------------

class TestBestActions:

    def test_shapes(self, scalar_gp):
        actions, mu, std = scalar_gp.best_actions()
        assert actions.shape == (1, 2)
        assert mu.shape == (1,) and std.shape == (1,)

    def test_action_is_in_the_raw_box(self, scalar_gp):
        actions, _, _ = scalar_gp.best_actions()
        assert np.all(actions >= LOW) and np.all(actions <= HIGH)

    def test_a_one_hot_weight_recovers_that_objectives_optimum(self, mogp):
        best_mo, _, _ = mogp.best_actions(raw_samples=1024)
        for j in range(len(mogp.objectives)):
            best_w, _, _ = mogp.scalarized(np.eye(2)[j]).best_actions(raw_samples=1024)
            assert np.allclose(best_w[0], best_mo[j], atol=1e-2)

    def test_values_match_the_posterior_there(self, scalar_gp):
        actions, mu, std = scalar_gp.best_actions()
        mu_at, std_at = scalar_gp.posterior_at(actions)
        assert np.allclose(mu, mu_at[:, 0]) and np.allclose(std, std_at[:, 0])


# ---- what the acquisition layer reads --------------------------------------

class TestAcquisitionInterface:

    def test_frame_is_the_shared_one(self, scalar_gp, objectives):
        assert scalar_gp.frame is objectives.xtransform

    def test_measured_x_is_the_union_of_every_objectives_actions(self, scalar_gp, actions):
        # both objectives are measured at the same grid here, so the union of
        # the two is that grid, deduplicated
        assert scalar_gp.measured_x.shape == actions.shape

    def test_measured_x_is_normalized(self, scalar_gp):
        assert np.all(scalar_gp.measured_x >= 0) and np.all(scalar_gp.measured_x <= 1)

    def test_incumbent_is_the_best_posterior_mean_over_measured_actions(self, scalar_gp):
        mu, _ = scalar_gp.posterior_at(scalar_gp.measured_x, normalized=True)
        assert scalar_gp.incumbent() == mu.max()

    def test_incumbent_is_at_most_the_best_action(self, scalar_gp):
        # best_actions searches the continuous box, the incumbent only the
        # measured actions, so the former can only be better
        _, best, _ = scalar_gp.best_actions(raw_samples=1024)
        assert scalar_gp.incumbent() <= best[0] + 1e-6
