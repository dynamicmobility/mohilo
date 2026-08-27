"""Tests for the hypervolume reference point.

The reference is the worst value per objective that still counts, so the
contract is about *direction*: it sits a margin past the bad end of every
objective, whichever end that is. The tests pin that a declared box and a set of
observations that fill it agree, that a maximized column pushes the other way,
and that a caller cannot state the range twice or state it in the wrong shape.
"""

import numpy as np
import pytest

from pypolar.utils.pareto import get_nondominated, reference_point

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
