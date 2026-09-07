"""Tests for the hypervolume reference point.

The reference is the worst value per objective that still counts, so the
contract is about *direction*: it sits a margin past the bad end of every
objective, whichever end that is. The tests pin that a declared box and a set of
observations that fill it agree, that a maximized column pushes the other way,
and that a caller cannot state the range twice or state it in the wrong shape.
"""

import numpy as np
import pytest

from pypolar.optimization.objectives import DecoupledObjectives, Objective
from pypolar.utils.pareto import (get_nondominated, hypervolume_from_objectives,
                                  reference_point, reference_point_from_objectives)

BOUNDS = np.array([[0.0, -1.0],      # low  per objective
                   [4.0,  1.0]])     # high per objective


class TestReferencePoint:

    def test_a_minimized_objective_sits_above_its_high_end(self):
        # margin 0.1 of ranges 4 and 2
        np.testing.assert_allclose(reference_point(bounds=BOUNDS),
                                   [4.4, 1.2])

    def test_a_maximized_objective_sits_below_its_low_end(self):
        np.testing.assert_allclose(
            reference_point(bounds=BOUNDS, maximize=[True, True]),
            [-0.4, -1.2]
        )

    def test_each_column_takes_its_own_direction(self):
        np.testing.assert_allclose(
            reference_point(bounds=BOUNDS, maximize=[True, False]),
            [-0.4, 1.2]
        )

    def test_the_margin_is_a_fraction_of_the_range(self):
        np.testing.assert_allclose(reference_point(bounds=BOUNDS, margin=0.0),
                                   BOUNDS[1])
        np.testing.assert_allclose(reference_point(bounds=BOUNDS, margin=0.5),
                                   [6.0, 2.0])

    def test_values_and_the_box_they_fill_agree(self):
        # the corners alone span the box, so reading the range off them is the
        # same measurement as declaring it
        corners = np.array([[0.0, -1.0], [4.0, -1.0], [0.0, 1.0], [4.0, 1.0]])

        np.testing.assert_allclose(reference_point(values=corners),
                                   reference_point(bounds=BOUNDS))

    def test_values_read_the_range_they_actually_cover(self):
        # a set that does not reach the box's corners states a smaller range,
        # so its reference lands inside the box's own
        values = np.array([[1.0, 0.0], [2.0, 0.5]])

        np.testing.assert_allclose(reference_point(values=values),
                                   [2.1, 0.55])

    def test_one_observation_is_a_degenerate_range(self):
        # zero range, so the reference is the point itself: nothing to be a
        # fraction of
        np.testing.assert_allclose(reference_point(values=[[1.0, 2.0]]),
                                   [1.0, 2.0])

    def test_no_point_of_the_set_is_worse_than_the_reference(self):
        # what the reference is for: every measurement counts, so every one of
        # them dominates it
        rng    = np.random.default_rng(0)
        values = rng.normal(size=(64, 3)) * [1.0, 10.0, 0.1]
        ref    = reference_point(values=values, maximize=[False, True, False])

        signs = np.array([-1.0, 1.0, -1.0])
        assert np.all(signs * values > signs * ref)

    def test_the_front_it_measures_is_the_sets_own(self):
        # a sanity check that the reference and get_nondominated read the same
        # direction: the front of a minimized set beats the reference
        values = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 2.0]])
        ref    = reference_point(values=values)
        front  = values[get_nondominated(-values)]

        assert len(front) == 3                      # (2, 2) is dominated
        assert np.all(front < ref)

    def test_stating_the_range_twice_is_an_error(self):
        with pytest.raises(ValueError):
            reference_point(values=[[1.0, 2.0]], bounds=BOUNDS)

    def test_stating_it_not_at_all_is_an_error(self):
        with pytest.raises(ValueError):
            reference_point()

    def test_a_box_is_two_rows_not_two_columns(self):
        # (m, 2) pairs of (low, high) would silently scramble the two rows
        with pytest.raises(ValueError):
            reference_point(bounds=[[0.0, 4.0], [-1.0, 1.0], [0.0, 1.0]])

    def test_a_direction_per_column_or_none(self):
        with pytest.raises(ValueError):
            reference_point(bounds=BOUNDS, maximize=[True])


class TestReferencePointFromObjectives:
    """The same reference, read off a `DecoupledObjectives`.

    The objectives are decoupled, so each carries its own measurements in its
    own number: the range is per objective rather than off one shared array,
    and each objective's own `maximize` picks the end to push past.
    """

    @staticmethod
    def make(values, maximize):
        # one action column, distinct per point
        return DecoupledObjectives([
            Objective.from_data(
                actions  = np.arange(len(v), dtype=float).reshape(-1, 1),
                values   = np.asarray(v, dtype=float),
                maximize = m,
                name     = f'obj{i}')
            for i, (v, m) in enumerate(zip(values, maximize))
        ])

    def test_it_matches_the_box_those_measurements_fill(self):
        # columns spanning BOUNDS, both minimized: the same answer the declared
        # box gives
        objectives = self.make([[0.0, 4.0], [-1.0, 1.0]], [False, False])

        np.testing.assert_allclose(
            reference_point_from_objectives(objectives),
            reference_point(bounds=BOUNDS)
        )

    def test_each_objective_takes_its_own_direction(self):
        objectives = self.make([[0.0, 4.0], [-1.0, 1.0]], [True, False])

        np.testing.assert_allclose(
            reference_point_from_objectives(objectives), [-0.4, 1.2])

    def test_the_margin_is_a_fraction_of_the_range(self):
        objectives = self.make([[0.0, 4.0], [-1.0, 1.0]], [False, False])

        np.testing.assert_allclose(
            reference_point_from_objectives(objectives, margin=0.0), [4.0, 1.0])
        np.testing.assert_allclose(
            reference_point_from_objectives(objectives, margin=0.5), [6.0, 2.0])

    def test_the_objectives_need_not_share_a_count(self):
        # the point of decoupling: three measurements on one objective and two
        # on the other, each read on its own
        objectives = self.make([[0.0, 2.0, 4.0], [-1.0, 1.0]], [False, False])

        np.testing.assert_allclose(
            reference_point_from_objectives(objectives), [4.4, 1.2])

    def test_no_measurement_is_worse_than_the_reference(self):
        rng        = np.random.default_rng(0)
        values     = [rng.normal(size=8), rng.normal(size=5) * 10.0]
        maximize   = [False, True]
        objectives = self.make(values, maximize)
        ref        = reference_point_from_objectives(objectives)

        for v, m, r in zip(values, maximize, ref):
            sign = 1.0 if m else -1.0
            assert np.all(sign * v > sign * r)

    def test_an_objective_with_no_measurements_is_an_error(self):
        objectives = DecoupledObjectives([
            Objective.from_data(actions=np.array([[0.0], [1.0]]),
                                values=np.array([0.0, 1.0]),
                                maximize=False, name='measured'),
            Objective.from_empty(name='declared', maximize=False),
        ])

        with pytest.raises(ValueError, match='declared'):
            reference_point_from_objectives(objectives)

    def test_no_objectives_at_all_is_an_error(self):
        with pytest.raises(ValueError):
            reference_point_from_objectives(DecoupledObjectives([]))


class TestHypervolumeFromObjectives:
    """The volume a `DecoupledObjectives`' measurements dominate.

    The tests pin the geometry on a hand-checkable front, that direction is
    read off each objective's own `maximize` rather than from the caller, that
    a measurement past the reference contributes nothing, and that values
    paired by position are refused when position does not mean the same action
    in every column.
    """

    # (1, 3), (3, 1), (2, 2): all three non-dominated when minimized, and
    # against the reference (4, 4) they cover 6 units of area
    VALUES = [[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]]
    REF    = [4.0, 4.0]

    @staticmethod
    def make(values, maximize):
        # every objective at the same actions, one column, distinct per point
        actions = np.arange(len(values[0]), dtype=float).reshape(-1, 1)
        return DecoupledObjectives([
            Objective.from_data(actions  = actions,
                                values   = np.asarray(v, dtype=float),
                                maximize = m,
                                name     = f'obj{i}')
            for i, (v, m) in enumerate(zip(values, maximize))
        ])

    def test_the_volume_between_the_front_and_the_reference(self):
        objectives = self.make(self.VALUES, [False, False])

        assert hypervolume_from_objectives(objectives, self.REF) == \
            pytest.approx(6.0)

    def test_a_dominated_measurement_adds_nothing(self):
        # (3, 3) is dominated by (2, 2), so it lies inside the volume already
        objectives = self.make([[1.0, 3.0, 2.0, 3.0], [3.0, 1.0, 2.0, 3.0]],
                               [False, False])

        assert hypervolume_from_objectives(objectives, self.REF) == \
            pytest.approx(6.0)

    def test_a_measurement_past_the_reference_adds_nothing(self):
        # (9, 0.5) is non-dominated -- it beats everything on the second
        # objective -- but its first value is worse than the reference, so its
        # box is empty
        objectives = self.make([[1.0, 3.0, 2.0, 9.0], [3.0, 1.0, 2.0, 0.5]],
                               [False, False])

        assert hypervolume_from_objectives(objectives, self.REF) == \
            pytest.approx(6.0)

    def test_direction_comes_from_the_objectives_not_the_caller(self):
        # the same geometry stated as a maximization: negating the values and
        # the reference together leaves the volume unchanged, because `signs`
        # flips it back
        flipped = self.make([[-v for v in col] for col in self.VALUES],
                            [True, True])

        assert hypervolume_from_objectives(flipped, [-4.0, -4.0]) == \
            pytest.approx(6.0)

    def test_a_mixed_pair_of_directions(self):
        # second objective maximized: (3, -1), (1, -3), (2, -2) against
        # (4, -4) is the same front, so the same volume
        mixed = self.make([self.VALUES[0], [-v for v in self.VALUES[1]]],
                          [False, True])

        assert hypervolume_from_objectives(mixed, [4.0, -4.0]) == \
            pytest.approx(6.0)

    def test_the_reference_it_pairs_with(self):
        # reference_point_from_objectives is the reference this is meant to be
        # measured against, and it leaves every measurement counting
        objectives = self.make(self.VALUES, [False, False])
        ref        = reference_point_from_objectives(objectives)

        # ranges 2 and 2, margin 0.1 -> (3.2, 3.2), so every box shrinks and
        # the union falls from 6.0 to 1.84
        np.testing.assert_allclose(ref, [3.2, 3.2])
        assert hypervolume_from_objectives(objectives, ref) == \
            pytest.approx(1.84)

    def test_objectives_at_different_actions_are_refused(self):
        objectives = DecoupledObjectives([
            Objective.from_data(actions=np.array([[0.0], [1.0]]),
                                values=np.array([1.0, 3.0]),
                                maximize=False, name='here'),
            Objective.from_data(actions=np.array([[0.0], [2.0]]),
                                values=np.array([3.0, 1.0]),
                                maximize=False, name='there'),
        ])

        with pytest.raises(ValueError, match='there'):
            hypervolume_from_objectives(objectives, self.REF)

    def test_objectives_with_different_counts_are_refused(self):
        objectives = DecoupledObjectives([
            Objective.from_data(actions=np.array([[0.0], [1.0], [2.0]]),
                                values=np.array([1.0, 3.0, 2.0]),
                                maximize=False, name='three'),
            Objective.from_data(actions=np.array([[0.0], [1.0]]),
                                values=np.array([3.0, 1.0]),
                                maximize=False, name='two'),
        ])

        with pytest.raises(ValueError, match='two'):
            hypervolume_from_objectives(objectives, self.REF)

    def test_a_reference_value_per_objective(self):
        objectives = self.make(self.VALUES, [False, False])

        with pytest.raises(ValueError):
            hypervolume_from_objectives(objectives, [4.0])

    def test_no_objectives_at_all_is_an_error(self):
        with pytest.raises(ValueError):
            hypervolume_from_objectives(DecoupledObjectives([]), self.REF)
