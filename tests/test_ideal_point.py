"""Tests for `IdealPoint`, the quadratic bowl with a declared optimum.

The claims worth pinning are that the formula is what the docstring says, that
the optimum really is the minimum of the box, that the botorch contract holds
(`bounds`, `dim`, `optimal_value`, the noiseless `noise=False` path), and that
several bowls compose into one `MOSyntheticOracle`.
"""

import json

import numpy as np
import pytest
import torch

from pypolar.feedback.synthetic import (IdealPoint, MOSyntheticOracle,
                                        SyntheticOracle, truth_at)
from pypolar.optimization.objectives import sample_actions


OPTIMUM = np.array([0.5, -0.25])
WEIGHTS = np.array([1.0, 4.0])
OFFSET  = 2.0


@pytest.fixture
def bowl():
    return IdealPoint(optimum=OPTIMUM, weights=WEIGHTS, offset=OFFSET)


def _closed_form(X, optimum=OPTIMUM, weights=WEIGHTS, offset=OFFSET):
    return np.sum(weights * (X - optimum) ** 2, axis=-1) + offset


def test_values_match_the_formula(bowl):
    X = sample_actions(bounds=bowl.bounds, n=64, kind='sobol', seed=0)
    assert np.allclose(truth_at(bowl, X), _closed_form(X))


def test_optimum_is_the_minimum(bowl):
    X = sample_actions(bounds=bowl.bounds, n=512, kind='sobol', seed=0)
    assert truth_at(bowl, OPTIMUM[None, :])[0] == pytest.approx(OFFSET)
    assert np.all(truth_at(bowl, X) >= OFFSET)


def test_optimal_value_and_optimizer_are_reported(bowl):
    assert bowl.optimal_value == pytest.approx(OFFSET)
    assert np.allclose(bowl.optimizers.numpy(), OPTIMUM[None, :])
    assert bowl.dim == 2


def test_negate_flips_the_reported_optimum():
    bowl = IdealPoint(optimum=OPTIMUM, offset=OFFSET, negate=True)
    assert bowl.optimal_value == pytest.approx(-OFFSET)
    assert truth_at(bowl, OPTIMUM[None, :])[0] == pytest.approx(-OFFSET)


@pytest.mark.parametrize('dim', [1, 2, 5])
def test_extends_to_any_dimension(dim):
    bowl = IdealPoint(optimum=0.25, dim=dim)
    X    = sample_actions(bounds=bowl.bounds, n=32, kind='sobol', seed=0)
    assert bowl.dim == dim
    assert X.shape == (32, dim)
    assert np.allclose(truth_at(bowl, X),
                       _closed_form(X, optimum=0.25, weights=1.0, offset=0.0))


def test_scalar_weights_are_shared_by_every_dimension():
    shared  = IdealPoint(optimum=[0.0, 0.0], weights=3.0)
    spelled = IdealPoint(optimum=[0.0, 0.0], weights=[3.0, 3.0])
    X = sample_actions(bounds=shared.bounds, n=32, kind='sobol', seed=0)
    assert np.allclose(truth_at(shared, X), truth_at(spelled, X))


def test_default_box_holds_the_optimum():
    # half-width 1 unless the optimum sits outside it, so bowls with nearby
    # optima share a box by default
    assert np.allclose(IdealPoint(optimum=[0.5]).bounds.numpy(), [[-1.0], [1.0]])
    assert np.allclose(IdealPoint(optimum=[3.0]).bounds.numpy(), [[-3.0], [3.0]])


def test_custom_box_excluding_the_optimum_raises():
    with pytest.raises(ValueError):
        IdealPoint(optimum=[5.0], bounds=[(-1.0, 1.0)])


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        IdealPoint(optimum=[0.0, 0.0], dim=3)
    with pytest.raises(ValueError):
        IdealPoint(optimum=[0.0, 0.0], weights=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        IdealPoint(optimum=np.zeros((2, 2)))


def test_noise_is_off_when_asked(bowl):
    X      = sample_actions(bounds=bowl.bounds, n=16, kind='sobol', seed=0)
    tensor = torch.as_tensor(X)
    noisy  = IdealPoint(optimum=OPTIMUM, weights=WEIGHTS, offset=OFFSET,
                        noise_std=1.0)
    assert np.allclose(noisy(tensor, noise=False).numpy(), _closed_form(X))
    assert not np.allclose(noisy(tensor, noise=True).numpy(), _closed_form(X))


def test_synthetic_oracle_finds_the_optimum(bowl):
    oracle = SyntheticOracle(truth=bowl, rel_noise_std=0.0, n_spread=4096)
    assert oracle.sample_min == pytest.approx(OFFSET, abs=1e-2)
    assert np.allclose(oracle.sample_argmin, OPTIMUM, atol=5e-2)


def test_several_bowls_compose_into_one_mo_oracle():
    bowls  = [IdealPoint(optimum=[-0.5, -0.5]), IdealPoint(optimum=[0.5, 0.5])]
    oracle = MOSyntheticOracle(truth=bowls, rel_noise_std=0.0, seed=0)

    assert len(oracle) == 2
    assert np.allclose(oracle.bounds, [[-1.0, -1.0], [1.0, 1.0]])

    X = sample_actions(bounds=oracle.bounds, n=32, kind='sobol', seed=0)
    values = oracle(X, noise=False)
    assert values.shape == (32, 2)
    for i, one in enumerate(bowls):
        assert np.allclose(values[:, i], truth_at(one, X))

    # each bowl is minimized at its own point, so the two columns disagree
    assert np.argmin(values[:, 0]) != np.argmin(values[:, 1])


class TestParams:
    """`SyntheticOracleParams(func='IdealPoint', optima=...)` -- the record a run
    stores, and the m bowls it rebuilds into."""

    def _params(self, **overrides):
        from pypolar.feedback.synthetic import SyntheticOracleParams
        return SyntheticOracleParams(**{
            'func'       : 'IdealPoint',
            'objectives' : ('A', 'B'),
            'optima'     : ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
            'dim'        : 3,
            'box'        : (0.0, 3.0),
        } | overrides)

    def test_optima_make_it_multi_objective(self):
        params = self._params()
        assert params.multi_objective
        oracle = params.build()
        assert isinstance(oracle, MOSyntheticOracle)
        assert len(oracle) == 2

    def test_every_bowl_shares_the_declared_box(self):
        oracle = self._params().build()
        # the intersection MOSyntheticOracle.bounds takes is the box itself,
        # so both optima stay inside the action space
        assert np.allclose(oracle.bounds, [[0.0] * 3, [3.0] * 3])
        for one, optimum in zip(oracle, ((1.0,) * 3, (2.0,) * 3)):
            assert np.allclose(one.truth.optimum.numpy(), optimum)
            assert np.allclose(one.truth.bounds.numpy(), [[0.0] * 3, [3.0] * 3])

    def test_each_column_is_minimized_at_its_own_optimum(self):
        oracle = self._params().build()
        values = oracle(np.array([[1.0] * 3, [2.0] * 3]), noise=False)
        assert np.allclose(np.diag(values), 0.0)
        assert np.all(np.diag(values[::-1]) > 0.0)

    def test_a_json_round_trip_rebuilds_the_same_truth(self):
        from dataclasses import asdict
        from pypolar.feedback.synthetic import SyntheticOracleParams
        params = self._params()
        # json reads every sequence back as a list, which must not change
        # the truth or make the frozen record unhashable
        replayed = SyntheticOracleParams(**json.loads(json.dumps(asdict(params))))
        assert replayed == params
        assert hash(replayed) == hash(params)

        X = sample_actions(bounds=[[0.0] * 3, [3.0] * 3], n=16, kind='sobol', seed=0)
        assert np.allclose(replayed.build()(X, noise=False),
                           params.build()(X, noise=False))

    def test_a_scalar_box_still_means_a_half_width(self):
        oracle = self._params(optima=((0.5,), (-0.5,)), objectives=('A', 'B'),
                              dim=1, box=2.0).build()
        assert np.allclose(oracle.bounds, [[-2.0], [2.0]])

    def test_malformed_params_raise(self):
        with pytest.raises(ValueError):
            self._params(func='Levy')            # only IdealPoint takes optima
        with pytest.raises(ValueError):
            self._params(box=None)               # the bowls need a shared box
        with pytest.raises(ValueError):
            self._params(objectives=('A',))      # one name per optimum
        with pytest.raises(ValueError):
            self._params(num_objectives=2)       # optima already state m
