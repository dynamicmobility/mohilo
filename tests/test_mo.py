"""Tests for the multi-objective metrics.

The module's contract is that a truth carries no direction -- BoTorch states
every one of its own in the minimizing sense -- and that which way each column
is optimized comes from the objectives. The tests below pin that the
all-minimizing case is unchanged, that a maximized problem scores its own front
as perfect rather than as worthless, and that the two cannot be mixed silently.
"""

import numpy as np
import pytest

from botorch.test_functions import multi_objective

from pypolar.feedback.synthetic import MOSyntheticOracle
from pypolar.optimization.objectives import DecoupledObjectives, Objective
from pypolar.performance.mo import (_attained_fraction, _signs,
                                    attained_hypervolume_regret,
                                    front_alignment_regret,
                                    normalized_hypervolume_regret)
from pypolar.utils.pareto import (get_nondominated, hypervolume_from_nondominated)

SEED     = 3
N_SPREAD = 256      # a Sobol scan small enough to keep the suite quick


@pytest.fixture
def oracle():
    """BraninCurrin, whose two columns BoTorch states minimizing."""
    return MOSyntheticOracle(multi_objective.BraninCurrin(), n_spread=N_SPREAD,
                             seed=SEED)


def make_objectives(oracle, maximize):
    """Objectives over the oracle's own scan, one per truth column."""
    return DecoupledObjectives([
        Objective.from_data(
            actions  = oracle.scan_actions[:8],
            values   = oracle.scan_values[:8, i],
            maximize = maximize,
            name     = f'y{i}'
        )
        for i in range(len(oracle))
    ])


class StubGP:
    """The `posterior_at` contract the metrics ask for, and the objectives they
    read the direction off. A real fit would only add noise to what is being
    tested."""

    def __init__(self, objectives, mu):
        self.objectives = objectives
        self._mu        = mu

    def posterior_at(self, action, **kwargs):
        return self._mu, np.zeros_like(self._mu)


class TestDirectionIsTheObjectives:

    def test_minimizing_reproduces_the_hardcoded_formula(self, oracle):
        # what the module computed before the direction was an argument
        front    = oracle.scan_values[get_nondominated(-oracle.scan_values)]
        expected = hypervolume_from_nondominated(front - oracle.ref_point)

        assert oracle.max_hypervolume() == expected
        assert oracle.sampled_max_hypervolume == expected
        np.testing.assert_allclose(oracle.max_hypervolume([-1.0, -1.0]), expected)

    def test_a_direction_is_measured_once_and_kept(self, oracle):
        assert oracle.max_hypervolume([1.0, 1.0]) is oracle.max_hypervolume([1.0, 1.0])

    def test_maximizing_is_a_different_hypervolume(self, oracle):
        # the same scan against the same reference, read the other way, so the
        # two measure different regions and neither is the other's complement
        assert oracle.max_hypervolume([1.0, 1.0]) > 0
        assert oracle.max_hypervolume([1.0, 1.0]) != oracle.max_hypervolume()

    def test_signs_come_from_the_objectives(self, oracle):
        np.testing.assert_array_equal(
            make_objectives(oracle, maximize=False).signs, [-1.0, -1.0])
        np.testing.assert_array_equal(
            make_objectives(oracle, maximize=True).signs, [1.0, 1.0])


class TestAttainedFraction:

    @pytest.mark.parametrize('maximize', [False, True])
    def test_the_scan_scores_itself_perfectly(self, oracle, maximize):
        # the denominator is the scan's own front, so the whole scan attains
        # exactly it -- whichever way the objectives are pointed
        signs = make_objectives(oracle, maximize).signs

        assert _attained_fraction(oracle.scan_values, oracle, signs) == 1.0

    def test_the_wrong_direction_attains_nothing(self, oracle):
        # the minimizing front scored as if it were maximized: every point is
        # worse than the reference in some column, so its hypervolume is zero.
        # This is the failure the direction argument exists to prevent
        front = oracle.scan_values[get_nondominated(-oracle.scan_values)]

        assert _attained_fraction(front, oracle, np.array([1.0, 1.0])) == 0.0


class TestRegrets:

    @pytest.mark.parametrize('maximize', [False, True])
    def test_the_true_front_scores_zero_regret(self, oracle, maximize):
        objectives = make_objectives(oracle, maximize)
        # a model that knows the truth exactly: the posterior mean *is* the
        # truth in maximization space, so its inferred front is the true one
        gp = StubGP(objectives, objectives.signs * oracle.scan_values)

        assert normalized_hypervolume_regret(gp, oracle) == pytest.approx(0.0)
        assert front_alignment_regret(gp, oracle) == pytest.approx(0.0)

    @pytest.mark.parametrize('maximize', [False, True])
    def test_gd_plus_is_finite_and_non_negative(self, oracle, maximize):
        objectives = make_objectives(oracle, maximize)
        # a model that has it exactly backwards, which is the case that put
        # ideal above nadir and made pymoo's normalization raise
        gp = StubGP(objectives, -objectives.signs * oracle.scan_values)

        alignment = front_alignment_regret(gp, oracle)

        assert np.isfinite(alignment) and alignment > 0

    @pytest.mark.parametrize('maximize', [False, True])
    def test_attained_regret_takes_the_objectives(self, oracle, maximize):
        objectives = make_objectives(oracle, maximize)

        regret = attained_hypervolume_regret(oracle.scan_actions, oracle,
                                             objectives)

        assert regret == pytest.approx(0.0)

    def test_direction_changes_the_score(self, oracle):
        # the same actions, read the two ways, are not the same measurement
        minimized = attained_hypervolume_regret(
            oracle.scan_actions[:32], oracle, make_objectives(oracle, False))
        maximized = attained_hypervolume_regret(
            oracle.scan_actions[:32], oracle, make_objectives(oracle, True))

        assert minimized != maximized


class TestMismatch:

    def test_too_few_objectives_for_the_truth_raises(self, oracle):
        objectives = make_objectives(oracle, maximize=False)[['y0']]

        with pytest.raises(ValueError, match='truth columns'):
            _signs(objectives, oracle)

    def test_a_mismatched_gp_raises_rather_than_scoring(self, oracle):
        objectives = make_objectives(oracle, maximize=False)[['y0']]
        gp         = StubGP(objectives, oracle.scan_values[:, :1])

        with pytest.raises(ValueError, match='truth columns'):
            normalized_hypervolume_regret(gp, oracle)

    def test_max_hypervolume_checks_its_own_signs(self, oracle):
        with pytest.raises(ValueError, match='signs'):
            oracle.max_hypervolume([-1.0])
