"""Tests for AcquisitionFunction.

The class binds a BoTorch acquisition to a search *box* rather than to a model,
so the same instance can be queried against any single-output wrapper -- a
`BoTorchGP`, or a `DecoupledMOGP` read through weights as a `ScalarizedGP`. What
is worth pinning: the box is honoured in raw units, the incumbent arguments are
supplied only when the acquisition names them, and a scalarized model reaches
the acquisition through its `posterior_transform` rather than through a refit.
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

from pypolar.feedback.acquisition import AcquisitionFunction
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

class TestConstruction:

    def test_takes_a_single_objectives_bounds(self, reward):
        acq = AcquisitionFunction(UCB, reward)
        assert acq.action_box.shape == (2, 2)
        assert np.array_equal(acq.action_box, np.stack([LOW, HIGH]))

    def test_takes_a_collections_bounds(self, objectives):
        acq = AcquisitionFunction(UCB, objectives)
        assert np.array_equal(acq.action_box, np.stack([LOW, HIGH]))

    def test_a_scalar_bound_broadcasts_over_the_dimensions(self, reward):
        acq = AcquisitionFunction(UCB, reward, bounds=(0.0, 1.0))
        assert np.array_equal(acq.action_box, np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_unpinned_bounds_raise(self, actions):
        unpinned = Objective.from_data(
            actions=actions, values=_bump(actions, PEAK), maximize=True, name='reward'
        )
        with pytest.raises(ValueError):
            AcquisitionFunction(UCB, unpinned)

    def test_one_unpinned_member_unpins_the_collection(self, reward, actions):
        unpinned = Objective.from_data(
            actions=actions, values=_bump(actions, VALLEY), maximize=False, name='cost'
        )
        with pytest.raises(ValueError):
            AcquisitionFunction(UCB, DecoupledObjectives([reward, unpinned]))


# ---- what reaches the acquisition ------------------------------------------

class TestIncumbent:

    def test_ucb_takes_no_incumbent(self, gp, reward):
        # a single-output model carries no posterior transform either
        assert AcquisitionFunction(UCB, reward)._incumbent(gp) == {}

    def test_ei_takes_the_best_measured_value(self, gp, reward):
        args = AcquisitionFunction(ExpectedImprovement, reward)._incumbent(gp)
        assert args == {'best_f': reward.standard_y.max()}

    def test_a_bound_keyword_is_left_to_the_partial(self, gp, reward):
        acqf = partial(ExpectedImprovement, best_f=0.0)
        assert AcquisitionFunction(acqf, reward)._incumbent(gp) == {}

    def test_qlognei_takes_the_measured_actions(self, gp, reward):
        args = AcquisitionFunction(qLogNoisyExpectedImprovement, reward)._incumbent(gp)
        assert set(args) == {'X_baseline'}
        assert torch.equal(args['X_baseline'],
                           torch.as_tensor(reward.normalized_x, dtype=torch.float64))

    def test_a_scalarized_model_passes_its_transform(self, mogp, objectives):
        model = mogp.scalarized([0.6, 0.4])
        args = AcquisitionFunction(QUCB, objectives)._incumbent(model)
        assert args['posterior_transform'] is model.transform

    def test_a_scalarized_incumbent_is_not_evaluated_for_ucb(self, mogp, objectives):
        # it costs a posterior over every measured action, and UCB has no use
        # for it, so building it lazily is load-bearing rather than cosmetic
        model = mogp.scalarized([0.6, 0.4])
        model.incumbent = lambda: pytest.fail('incumbent evaluated for a UCB')
        AcquisitionFunction(QUCB, objectives)._incumbent(model)

    def test_a_scalarized_ei_takes_the_posterior_mean_incumbent(self, mogp, objectives):
        model = mogp.scalarized([0.6, 0.4])
        args = AcquisitionFunction(ExpectedImprovement, objectives)._incumbent(model)
        assert args['best_f'] == model.incumbent()


# ---- the query -------------------------------------------------------------

class TestQuery:

    def test_returns_one_action_in_the_raw_box(self, gp, reward):
        action = AcquisitionFunction(UCB, reward).query(gp)
        assert action.shape == (1, 2)
        assert _in_box(action)

    def test_normalized_actions_are_in_the_unit_box(self, gp, reward):
        action = AcquisitionFunction(UCB, reward).query(gp, raw=False)
        assert np.all(action >= 0) and np.all(action <= 1)

    def test_a_narrower_box_is_honoured(self, gp, reward):
        box = (np.array([0.0, -2.0]), np.array([2.0, 0.0]))
        action = AcquisitionFunction(UCB, reward, bounds=box).query(gp)
        assert np.all(action >= box[0] - 1e-9) and np.all(action <= box[1] + 1e-9)

    def test_ucb_finds_the_peak(self, gp, reward):
        # a smooth unimodal objective densely measured, so exploitation wins
        action = AcquisitionFunction(UCB, reward, raw_samples=1024).query(gp)
        assert np.allclose(action[0], PEAK, atol=1.0)


class TestQueryScalarized:

    def test_a_q_batch_is_returned_in_the_raw_box(self, mogp, objectives):
        model = mogp.scalarized([0.6, 0.4])
        actions = AcquisitionFunction(QUCB, objectives).query(model, q=4)
        assert actions.shape == (4, 2)
        assert _in_box(actions)

    def test_a_q_batch_is_not_all_one_point(self, mogp, objectives):
        torch.manual_seed(0)
        model = mogp.scalarized([0.6, 0.4])
        actions = AcquisitionFunction(QUCB, objectives).query(model, q=4)
        assert len(np.unique(actions.round(6), axis=0)) > 1

    def test_a_one_hot_weight_matches_that_objectives_own_gp(self, mogp, objectives, gp, reward):
        # 'reward' is objective 0, and gp is a BoTorchGP over it alone
        model = mogp.scalarized([1.0, 0.0])
        from_mogp = AcquisitionFunction(UCB, objectives, raw_samples=1024).query(model)
        from_gp   = AcquisitionFunction(UCB, reward,     raw_samples=1024).query(gp)
        assert np.allclose(from_mogp, from_gp, atol=1e-2)
