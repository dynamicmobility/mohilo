"""Tests for AcquisitionFunction.

The class carries a BoTorch acquisition and nothing about the problem, so the
same instance can be queried against any single-output wrapper -- a `BoTorchGP`,
or a `DecoupledMOGP` read through weights as a `ScalarizedGP`. What is worth
pinning: the box comes off the queried model and is honoured in raw units, the
incumbent arguments are supplied only when the acquisition names them, and a
scalarized model reaches the acquisition through its `posterior_transform`
rather than through a refit.
"""

from functools import partial

import numpy as np
import pytest
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

from pypolar.feedback.acquisition import AcquisitionFunction, acquisition_factory_1d
from pypolar.optimization.gp import BoTorchGP, DecoupledMOGP
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

# Raw action box, deliberately neither unit nor centered, so a query that
# forgets the normalizing xtransform lands outside it visibly.
LOW  = np.array([0.0, -2.0])
HIGH = np.array([10.0, 2.0])

PEAK   = np.array([7.5, 1.0])
VALLEY = np.array([2.5, -1.0])

UCB  = partial(UpperConfidenceBound,  beta=2.0)
QUCB = partial(qUpperConfidenceBound, beta=2.0)


def _grid(n=8):
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(LOW, HIGH)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def _bump(X, center, width=3.0):
    return np.exp(-np.sum((X - center) ** 2, axis=1) / width ** 2)


@pytest.fixture
def actions():
    return _grid()


@pytest.fixture
def reward(actions):
    return Objective.from_data(
        actions=actions, values=_bump(actions, PEAK), maximize=True,
        name='reward', action_bounds=(LOW, HIGH)
    )


@pytest.fixture
def cost(actions):
    return Objective.from_data(
        actions=actions, values=-_bump(actions, VALLEY), maximize=False,
        name='cost', action_bounds=(LOW, HIGH)
    )


@pytest.fixture
def objectives(reward, cost):
    return DecoupledObjectives([reward, cost])


@pytest.fixture
def gp(reward):
    return BoTorchGP(reward, noise=0.05, fit_hyperparameters=True)


@pytest.fixture
def mogp(objectives):
    return DecoupledMOGP(objectives, fit_hyperparameters=True)


def _in_box(actions):
    return np.all(actions >= LOW - 1e-9) and np.all(actions <= HIGH + 1e-9)


# ---- the box ---------------------------------------------------------------

def _unpinned_gp(actions, center=PEAK):
    """A GP over an objective that pins no action_bounds."""
    return BoTorchGP(
        Objective.from_data(actions=actions, values=_bump(actions, center),
                            maximize=True, name='reward'),
        noise=0.05, fit_hyperparameters=True
    )


class TestTheBox:

    def test_defaults_to_the_models_own_bounds(self, gp):
        assert _in_box(AcquisitionFunction(UCB).query(gp))

    def test_a_collections_bounds_reach_a_scalarized_model(self, mogp):
        assert _in_box(AcquisitionFunction(UCB).query(mogp.scalarized([0.6, 0.4])))

    def test_a_scalar_bound_broadcasts_over_the_dimensions(self, gp):
        # one pair of numbers against a 2D action, so both dimensions get [0, 1]
        action = AcquisitionFunction(UCB, bounds=(0.0, 1.0)).query(gp)
        assert np.all(action >= -1e-9) and np.all(action <= 1 + 1e-9)

    def test_unpinned_bounds_raise_at_query(self, actions):
        with pytest.raises(ValueError, match='action_bounds'):
            AcquisitionFunction(UCB).query(_unpinned_gp(actions))

    def test_explicit_bounds_stand_in_for_unpinned_ones(self, actions):
        box = (np.array([0.0, -2.0]), np.array([2.0, 0.0]))
        action = AcquisitionFunction(UCB, bounds=box).query(_unpinned_gp(actions))

        assert np.all(action >= box[0] - 1e-9) and np.all(action <= box[1] + 1e-9)

    def test_one_unpinned_member_unpins_the_collection(self, reward, actions):
        unpinned = Objective.from_data(
            actions=actions, values=_bump(actions, VALLEY), maximize=False, name='cost'
        )
        mogp = DecoupledMOGP(DecoupledObjectives([reward, unpinned]),
                             fit_hyperparameters=True)

        with pytest.raises(ValueError, match='action_bounds'):
            AcquisitionFunction(UCB).query(mogp.scalarized([1.0, 0.0]))


# ---- what reaches the acquisition ------------------------------------------

class TestIncumbent:

    def test_ucb_takes_no_incumbent(self, gp):
        # a single-output model carries no posterior transform either
        assert AcquisitionFunction(UCB)._incumbent(gp) == {}

    def test_ei_takes_the_best_measured_value(self, gp, reward):
        args = AcquisitionFunction(ExpectedImprovement)._incumbent(gp)
        assert args == {'best_f': reward.standard_y.max()}

    def test_a_bound_keyword_is_left_to_the_partial(self, gp):
        acqf = partial(ExpectedImprovement, best_f=0.0)
        assert AcquisitionFunction(acqf)._incumbent(gp) == {}

    def test_qlognei_takes_the_measured_actions(self, gp, reward):
        args = AcquisitionFunction(qLogNoisyExpectedImprovement)._incumbent(gp)
        assert set(args) == {'X_baseline'}
        assert torch.equal(args['X_baseline'],
                           torch.as_tensor(reward.normalized_x, dtype=torch.float64))

    def test_a_scalarized_model_passes_its_transform(self, mogp):
        model = mogp.scalarized([0.6, 0.4])
        args = AcquisitionFunction(QUCB)._incumbent(model)
        assert args['posterior_transform'] is model.transform

    def test_a_scalarized_incumbent_is_not_evaluated_for_ucb(self, mogp):
        # it costs a posterior over every measured action, and UCB has no use
        # for it, so building it lazily is load-bearing rather than cosmetic
        model = mogp.scalarized([0.6, 0.4])
        model.incumbent = lambda: pytest.fail('incumbent evaluated for a UCB')
        AcquisitionFunction(QUCB)._incumbent(model)

    def test_a_scalarized_ei_takes_the_posterior_mean_incumbent(self, mogp):
        model = mogp.scalarized([0.6, 0.4])
        args = AcquisitionFunction(ExpectedImprovement)._incumbent(model)
        assert args['best_f'] == model.incumbent()


# ---- the query -------------------------------------------------------------

class TestQuery:

    def test_returns_one_action_in_the_raw_box(self, gp):
        action = AcquisitionFunction(UCB).query(gp)
        assert action.shape == (1, 2)
        assert _in_box(action)

    def test_normalized_actions_are_in_the_unit_box(self, gp):
        action = AcquisitionFunction(UCB).query(gp, raw=False)
        assert np.all(action >= 0) and np.all(action <= 1)

    def test_a_narrower_box_is_honoured(self, gp):
        box = (np.array([0.0, -2.0]), np.array([2.0, 0.0]))
        action = AcquisitionFunction(UCB, bounds=box).query(gp)
        assert np.all(action >= box[0] - 1e-9) and np.all(action <= box[1] + 1e-9)

    def test_ucb_finds_the_peak(self, gp):
        # a smooth unimodal objective densely measured, so exploitation wins
        action = AcquisitionFunction(UCB, raw_samples=1024).query(gp)
        assert np.allclose(action[0], PEAK, atol=1.0)


class TestQueryScalarized:

    def test_a_q_batch_is_returned_in_the_raw_box(self, mogp):
        model = mogp.scalarized([0.6, 0.4])
        actions = AcquisitionFunction(QUCB).query(model, q=4)
        assert actions.shape == (4, 2)
        assert _in_box(actions)

    def test_a_q_batch_is_not_all_one_point(self, mogp):
        torch.manual_seed(0)
        model = mogp.scalarized([0.6, 0.4])
        actions = AcquisitionFunction(QUCB).query(model, q=4)
        assert len(np.unique(actions.round(6), axis=0)) > 1

    def test_a_one_hot_weight_matches_that_objectives_own_gp(self, mogp, gp):
        # 'reward' is objective 0, and gp is a BoTorchGP over it alone
        model = mogp.scalarized([1.0, 0.0])
        from_mogp = AcquisitionFunction(UCB, raw_samples=1024).query(model)
        from_gp   = AcquisitionFunction(UCB,     raw_samples=1024).query(gp)
        assert np.allclose(from_mogp, from_gp, atol=1e-2)


# ---- thompson sampling -----------------------------------------------------

class TestThompsonSampling:
    """`'ts'` is the one strategy whose acquisition is a *draw* rather than a
    deterministic functional of the posterior: `PathwiseThompsonSampling` draws
    a Matheron path, caches it, and `optimize_acqf` maximizes that one path. So
    what is worth pinning is the randomness -- fresh per query, seedable, and
    conditioned on the measurements rather than on the prior.
    """

    def test_the_factory_returns_the_pathwise_class(self):
        # bare, not a partial: TS takes no knob the factory could bind
        assert acquisition_factory_1d('ts', seed=0) is PathwiseThompsonSampling

    def test_the_factory_ignores_the_seed(self):
        # the path comes from torch's global generator, so two seeds cannot
        # give two different acquisitions
        assert acquisition_factory_1d('ts', seed=0) is acquisition_factory_1d('ts', seed=1)

    def test_takes_no_incumbent(self, gp):
        # no best_f, no baseline: a path is maximized as if it were the truth
        assert AcquisitionFunction(PathwiseThompsonSampling)._incumbent(gp) == {}

    def test_returns_one_action_in_the_raw_box(self, gp):
        action = AcquisitionFunction(PathwiseThompsonSampling).query(gp)
        assert action.shape == (1, 2)
        assert _in_box(action)

    def test_a_q_batch_is_returned_and_is_not_all_one_point(self, gp):
        # q independent paths, summed over the q-batch; the sum separates, so
        # maximizing it jointly is maximizing each path alone
        torch.manual_seed(0)
        actions = AcquisitionFunction(PathwiseThompsonSampling).query(gp, q=3)
        assert actions.shape == (3, 2)
        assert _in_box(actions)
        assert len(np.unique(actions.round(4), axis=0)) == 3

    def test_successive_queries_draw_different_paths(self, gp):
        # the defining property: a fresh path per query, where UCB below is a
        # fixed function of the posterior and repeats itself exactly
        acq = AcquisitionFunction(PathwiseThompsonSampling)
        torch.manual_seed(0)
        draws = np.vstack([acq.query(gp) for _ in range(4)])
        assert draws.std(axis=0).max() > 1e-3

        ucb = AcquisitionFunction(UCB)
        assert np.allclose(ucb.query(gp), ucb.query(gp), atol=1e-6)

    def test_manual_seed_makes_a_query_repeatable(self, gp):
        acq = AcquisitionFunction(PathwiseThompsonSampling)
        torch.manual_seed(0)
        first = acq.query(gp)
        torch.manual_seed(0)
        assert np.allclose(first, acq.query(gp), atol=1e-6)

    def test_the_path_is_conditioned_on_the_measurements(self, gp):
        # a smooth bump measured densely at low noise leaves a tight posterior,
        # so every path's argmax sits near the peak; a draw from the *prior*
        # would scatter over the whole box
        torch.manual_seed(0)
        acq = AcquisitionFunction(PathwiseThompsonSampling, raw_samples=1024)
        draws = np.vstack([acq.query(gp) for _ in range(4)])
        assert np.allclose(draws, PEAK, atol=1.0)
