"""Tests for the multi-objective metrics.

The module's contract is that a truth carries no direction -- BoTorch states
every one of its own in the minimizing sense -- and that which way each column
is optimized comes from the objectives. Nothing here needs a model: the metrics
take objective vectors and actions, so the tests below feed them directly.
They pin that
the all-minimizing case scores the scan's own front as perfect, that a
maximized problem does too rather than scoring it as worthless, that the
numerator and the denominator share one reference, and that the two directions
cannot be mixed silently.
"""

import numpy as np
import pytest

from botorch.test_functions import multi_objective

from pypolar.feedback.synthetic import MOSyntheticOracle
from pypolar.optimization.objectives import DecoupledObjectives, Objective
from pypolar.performance.mo import (_attained_fraction,
                                    attained_hypervolume_regret,
                                    front_alignment_regret,
                                    normalized_hypervolume_regret)
from pypolar.utils.pareto import (get_nondominated, hypervolume_from_nondominated,
                                  reference_point)

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


class TestDirectionIsTheObjectives:

    def test_minimizing_reproduces_the_hardcoded_formula(self, oracle):
        # what the module computed when the direction was a sign array: the
        # scan's own minimizing front, measured from a reference below it
        objectives = make_objectives(oracle, maximize=False)
        ref        = reference_point(values=oracle.scan_values,
                                     maximize=[False, False])

        front    = oracle.scan_values[get_nondominated(-oracle.scan_values)]
        expected = hypervolume_from_nondominated(front - ref)

        assert oracle.max_hypervolume(objectives) == pytest.approx(expected)

    def test_a_direction_is_measured_once_and_kept(self, oracle):
        objectives = make_objectives(oracle, maximize=True)

        assert (oracle.max_hypervolume(objectives)
                is oracle.max_hypervolume(objectives))

    def test_maximizing_is_a_different_hypervolume(self, oracle):
        # the same scan against its own reference, read the other way, so the
        # two measure different regions and neither is the other's complement
        maximized = oracle.max_hypervolume(make_objectives(oracle, True))
        minimized = oracle.max_hypervolume(make_objectives(oracle, False))

        assert maximized > 0
        assert maximized != minimized

    def test_the_map_is_sign_only(self, oracle):
        # no scale and no shift, so it does not move as measurements arrive and
        # the values stay in the objectives' own units
        objectives = make_objectives(oracle, maximize=False)

        np.testing.assert_allclose(
            objectives.maximization_space(oracle.scan_values),
            -oracle.scan_values)
        np.testing.assert_allclose(
            make_objectives(oracle, True).maximization_space(oracle.scan_values),
            oracle.scan_values)

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
        objectives = make_objectives(oracle, maximize)

        assert _attained_fraction(oracle.scan_values, objectives,
                                  oracle) == pytest.approx(1.0)

    def test_the_wrong_direction_scores_far_worse(self, oracle):
        # the direction reaches the reference as well as the values, so a set
        # read the wrong way up is scored self-consistently rather than being
        # measured from a reference on the wrong side. It is still nearly
        # worthless -- the minimizing front is a poor maximizing one -- which
        # is what the direction argument exists to make visible
        front = oracle.scan_values[get_nondominated(-oracle.scan_values)]

        assert _attained_fraction(front, make_objectives(oracle, False),
                                  oracle) == pytest.approx(1.0)
        assert _attained_fraction(front, make_objectives(oracle, True),
                                  oracle) < 0.1

    def test_one_reference_covers_both_halves(self, oracle):
        # numerator and denominator are measured from the same reference, so a
        # reference the caller supplies moves the ratio only where the set is
        # short of the front -- not by rescaling one half of it
        objectives = make_objectives(oracle, maximize=False)
        ref        = reference_point(values=oracle.scan_values,
                                     maximize=[False, False], margin=0.5)

        assert _attained_fraction(oracle.scan_values, objectives, oracle,
                                  ref_point=ref) == pytest.approx(1.0)


class TestRegrets:

    @pytest.mark.parametrize('maximize', [False, True])
    def test_the_true_front_scores_zero_regret(self, oracle, maximize):
        objectives = make_objectives(oracle, maximize)

        assert normalized_hypervolume_regret(
            oracle.scan_actions, oracle, objectives) == pytest.approx(0.0)

    def test_a_partial_front_scores_worse_than_the_whole(self, oracle):
        objectives = make_objectives(oracle, maximize=False)

        whole = normalized_hypervolume_regret(oracle.scan_actions, oracle,
                                              objectives)
        part  = normalized_hypervolume_regret(oracle.scan_actions[:16], oracle,
                                              objectives)

        assert 0.0 <= whole < part <= 1.0

    @pytest.mark.parametrize('maximize', [False, True])
    def test_the_observation_noise_cannot_flatter_a_run(self, oracle, maximize):
        """The actions are scored on the truth, so how noisy the oracle is
        cannot move the number -- and the regret stays in [0, 1]. Scored on the
        values a run claims instead, a lucky draw reports a point as better than
        anything the box holds and the regret comes out negative."""
        objectives = make_objectives(oracle, maximize)
        noisy      = MOSyntheticOracle(multi_objective.BraninCurrin(),
                                       rel_noise_std=0.5, n_spread=N_SPREAD,
                                       seed=SEED)

        clean_regret = normalized_hypervolume_regret(oracle.scan_actions[:32],
                                                     oracle, objectives)
        noisy_regret = normalized_hypervolume_regret(noisy.scan_actions[:32],
                                                     noisy, objectives)

        assert clean_regret == pytest.approx(noisy_regret)
        assert 0.0 <= clean_regret <= 1.0

    def test_the_two_hypervolume_regrets_are_one_measurement(self, oracle):
        # they differ only in what is handed to them: the front a run would
        # recommend, or every action it queried
        objectives = make_objectives(oracle, maximize=False)

        assert normalized_hypervolume_regret(
            oracle.scan_actions[:32], oracle, objectives
        ) == attained_hypervolume_regret(
            oracle.scan_actions[:32], oracle, objectives)

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

    @pytest.mark.parametrize('maximize', [False, True])
    def test_gd_plus_is_zero_on_the_true_front(self, oracle, maximize):
        # the actions a run that knew the truth exactly would nominate: the
        # scan's own front, which is the front the indicator measures against
        objectives = make_objectives(oracle, maximize)
        values     = objectives.maximization_space(oracle.scan_values)
        front      = get_nondominated(values)

        assert front_alignment_regret(oracle.scan_actions[front], oracle,
                                      objectives) == pytest.approx(0.0)

    @pytest.mark.parametrize('maximize', [False, True])
    def test_gd_plus_is_finite_and_non_negative(self, oracle, maximize):
        # a run that has it exactly backwards nominates the anti-front, which
        # is the case that put ideal above nadir and made pymoo's normalization
        # raise
        objectives = make_objectives(oracle, maximize)
        anti       = get_nondominated(
            -objectives.maximization_space(oracle.scan_values))

        alignment = front_alignment_regret(oracle.scan_actions[anti], oracle,
                                           objectives)

        assert np.isfinite(alignment) and alignment > 0


class TestMismatch:

    def test_too_few_objectives_for_the_truth_raises(self, oracle):
        objectives = make_objectives(oracle, maximize=False)[['y0']]

        with pytest.raises(ValueError, match='truth columns'):
            oracle.max_hypervolume(objectives)

    def test_a_mismatched_set_of_objectives_raises_rather_than_scoring(self, oracle):
        objectives = make_objectives(oracle, maximize=False)[['y0']]

        with pytest.raises(ValueError, match='objectives'):
            normalized_hypervolume_regret(oracle.scan_actions, oracle, objectives)
