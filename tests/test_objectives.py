"""Tests for AffineTransform, Objective and DecoupledObjectives.

Each test states one property in the terms the class name promises: an affine
transform is invertible, a "centered" transform removes the mean, a
"normalized" transform maps data onto [0, 1], and a set of *decoupled*
objectives lets each objective carry its own measurements, in its own number,
over one shared action frame.
"""

import numpy as np
import pytest
import torch
from botorch.test_functions import Levy, StyblinskiTang

from pypolar.optimization.objectives import (
    AffineTransform,
    DecoupledObjectives,
    Objective,
    sample_actions,
)


# ---- fixtures --------------------------------------------------------------

@pytest.fixture
def y():
    """Five scalar measurements, mean 6.0, min 2.0, max 10.0."""
    return np.array([2.0, 4.0, 6.0, 8.0, 10.0])


@pytest.fixture
def x():
    """Five actions in 3 dimensions. Column ranges are 4, 8 and 0.4."""
    return np.array([
        [0.0, 0.0, 0.1],
        [1.0, 2.0, 0.2],
        [2.0, 4.0, 0.3],
        [3.0, 6.0, 0.4],
        [4.0, 8.0, 0.5],
    ])


@pytest.fixture
def objective(y, x):
    return Objective(name='cost', maximize=False, ydata=y.copy(), xdata=x.copy())


@pytest.fixture
def pair(y, x):
    """Two objectives over the same five actions, pointing opposite ways."""
    return DecoupledObjectives([
        Objective(name='cost',    maximize=False, ydata=y.copy(),       xdata=x.copy()),
        Objective(name='comfort', maximize=True,  ydata=y.copy() * 2.0, xdata=x.copy()),
    ])


@pytest.fixture
def ragged(y, x):
    """The decoupled case: 'comfort' was measured three times, 'cost' five, and
    comfort's actions cover only the lower half of cost's action ranges."""
    return DecoupledObjectives([
        Objective(name='cost',    maximize=False, ydata=y.copy(),           xdata=x.copy()),
        Objective(name='comfort', maximize=True,  ydata=y[:3].copy() * 2.0, xdata=x[:3].copy()),
    ])


# ---- sample_actions --------------------------------------------------------

class TestSampleActions:
    """`bounds` is a (2, d) torch tensor of [lower; upper] rows, as BoTorch
    states them."""

    @pytest.fixture
    def bounds(self):
        """A box whose two dimensions have different ranges, 10 and 2."""
        return torch.tensor([[-5.0, 0.0], [5.0, 2.0]], dtype=torch.float64)

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_shape_is_n_by_dim(self, kind, dim):
        bounds = torch.stack([torch.zeros(dim, dtype=torch.float64),
                              torch.ones(dim, dtype=torch.float64)])
        assert sample_actions(bounds, 7, kind, 0).shape == (7, dim)

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_actions_are_float64(self, bounds, kind):
        assert sample_actions(bounds, 5, kind, 0).dtype == np.float64

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_actions_stay_inside_the_box(self, bounds, kind):
        actions = sample_actions(bounds, 256, kind, 0)
        lo, hi = bounds.numpy()
        assert np.all(actions >= lo) and np.all(actions <= hi)

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_each_dimension_gets_its_own_range(self, bounds, kind):
        # a box of 10 x 2, so ignoring the per-dimension bounds would overshoot
        actions = sample_actions(bounds, 256, kind, 0)
        assert actions[:, 0].min() < 0.0 < actions[:, 0].max()
        assert actions[:, 1].max() <= 2.0

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_the_same_seed_gives_the_same_actions(self, bounds, kind):
        assert sample_actions(bounds, 8, kind, 5) == pytest.approx(
            sample_actions(bounds, 8, kind, 5))

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_a_different_seed_gives_different_actions(self, bounds, kind):
        assert not np.allclose(sample_actions(bounds, 8, kind, 5),
                               sample_actions(bounds, 8, kind, 6))

    def test_the_two_designs_differ(self, bounds):
        assert not np.allclose(sample_actions(bounds, 8, 'sobol', 0),
                               sample_actions(bounds, 8, 'uniform', 0))

    @pytest.mark.parametrize('n', [8, 16, 32])
    def test_sobol_is_space_filling(self, bounds, n):
        """A Sobol sequence of 2^k points splits every axis exactly in half;
        that stratification is what "space-filling" buys over iid sampling."""
        actions = sample_actions(bounds, n, 'sobol', 3)
        midpoint = bounds.numpy().mean(axis=0)
        below = (actions < midpoint).sum(axis=0)
        assert below == pytest.approx(np.full(2, n // 2))

    @pytest.mark.parametrize('kind', ['not a design', 'sobal', 'Sobol', '', None])
    def test_an_unknown_kind_raises(self, bounds, kind):
        # the branch is a whitelist, so a typo cannot silently return a
        # different design than the one asked for
        with pytest.raises(ValueError):
            sample_actions(bounds, 6, kind, 1)

    @pytest.mark.parametrize('kind', ['sobol', 'uniform'])
    def test_bounds_may_be_any_array_like(self, bounds, kind):
        # coerced to a float64 tensor, so a list and a numpy array give the
        # same design as the tensor
        expected = sample_actions(bounds, 4, kind, 0)
        assert sample_actions(bounds.numpy(), 4, kind, 0) == pytest.approx(expected)
        assert sample_actions(bounds.tolist(), 4, kind, 0) == pytest.approx(expected)


# ---- AffineTransform -------------------------------------------------------

class TestAffineTransformCall:

    def test_applies_shift_then_scale(self):
        # (data + shift) * scale
        t = AffineTransform(scale=2.0, shift=1.0)
        assert t(3.0) == pytest.approx(8.0)

    def test_broadcasts_over_arrays(self):
        t = AffineTransform(scale=2.0, shift=1.0)
        got = t(np.array([0.0, 3.0]))
        assert got == pytest.approx(np.array([2.0, 8.0]))

    def test_zero_scale_is_rejected(self):
        # a zero scale collapses every input to the same value, so it cannot
        # be inverted
        with pytest.raises(ValueError):
            AffineTransform(scale=0.0, shift=1.0)

    def test_zero_in_a_vector_scale_is_rejected(self):
        with pytest.raises(ValueError):
            AffineTransform(scale=np.array([1.0, 0.0]), shift=0.0)


class TestAffineTransformInverse:

    def test_inv_undoes_call(self):
        t = AffineTransform(scale=2.0, shift=1.0)
        assert t.inv(t(3.0)) == pytest.approx(3.0)

    def test_call_undoes_inv(self):
        t = AffineTransform(scale=2.0, shift=1.0)
        assert t(t.inv(3.0)) == pytest.approx(3.0)

    def test_round_trip_on_an_array(self, y):
        t = AffineTransform(scale=0.5, shift=-3.0)
        assert t.inv(t(y)) == pytest.approx(y)


class TestAffineTransformInvScale:
    """A spread transforms by |scale| alone: mean and standard deviation do not
    transform the same way, so inv() is only correct for the mean."""

    def test_the_shift_is_ignored(self):
        t = AffineTransform(scale=2.0, shift=100.0)
        assert t.inv_scale(6.0) == pytest.approx(3.0)

    def test_a_negative_scale_still_gives_a_positive_spread(self):
        t = AffineTransform(scale=-2.0, shift=0.0)
        assert t.inv_scale(6.0) == pytest.approx(3.0)

    def test_it_matches_the_spread_of_the_inverted_samples(self, y):
        # SD[aY + b] = |a| SD[Y]: inverting the samples and taking their std is
        # the same as inv_scale of the samples' std
        t = AffineTransform.make_standardized(y, sign=-1.0)
        z = t(y)
        assert t.inv_scale(z.std()) == pytest.approx(t.inv(z).std())


class TestMakeCentered:

    def test_centered_data_has_zero_mean(self, y):
        t = AffineTransform.make_centered(y)
        assert t(y).mean() == pytest.approx(0.0)

    def test_scale_is_untouched(self, y):
        # centering only subtracts the mean; spacing between points is kept
        t = AffineTransform.make_centered(y)
        assert np.diff(t(y)) == pytest.approx(np.diff(y))

    def test_per_column_when_axis_is_given(self, x):
        t = AffineTransform.make_centered(x, axis=0)
        assert t(x).mean(axis=0) == pytest.approx(np.zeros(x.shape[1]))

    def test_an_explicit_scale_is_applied_after_centering(self, y):
        # (data - mean) * scale, so the spread is scaled but the centre stays
        t = AffineTransform.make_centered(y, scale=2.0)
        assert t(y) == pytest.approx(2.0 * (y - y.mean()))
        assert t(y).mean() == pytest.approx(0.0)

    def test_a_negative_scale_flips_the_data(self, y):
        # how a minimized objective is oriented without also standardizing it
        t = AffineTransform.make_centered(y, scale=-1.0)
        assert t(y) == pytest.approx(-(y - y.mean()))

    def test_a_zero_scale_is_rejected(self, y):
        with pytest.raises(ValueError):
            AffineTransform.make_centered(y, scale=0.0)


class TestMakeStandardized:

    def test_standardized_data_has_zero_mean_and_unit_std(self, y):
        t = AffineTransform.make_standardized(y)
        assert t(y).mean() == pytest.approx(0.0)
        assert t(y).std() == pytest.approx(1.0)

    def test_a_negative_sign_flips_the_data_but_not_its_spread(self, y):
        t = AffineTransform.make_standardized(y, sign=-1.0)
        assert t(y) == pytest.approx(-(y - y.mean()) / y.std())
        assert t(y).std() == pytest.approx(1.0)

    def test_constant_data_falls_back_to_centering(self):
        # std == 0 would divide by zero, so the scale is left at 1
        t = AffineTransform.make_standardized(np.full(4, 7.0))
        assert t(np.full(4, 7.0)) == pytest.approx(np.zeros(4))

    def test_per_column_when_axis_is_given(self, x):
        t = AffineTransform.make_standardized(x, axis=0)
        assert t(x).mean(axis=0) == pytest.approx(np.zeros(x.shape[1]))
        assert t(x).std(axis=0) == pytest.approx(np.ones(x.shape[1]))

    def test_one_constant_column_does_not_spoil_the_others(self, x):
        data = np.column_stack([x[:, 0], np.full(x.shape[0], 7.0)])
        t = AffineTransform.make_standardized(data, axis=0)
        assert t(data)[:, 0].std() == pytest.approx(1.0)
        assert t(data)[:, 1] == pytest.approx(np.zeros(x.shape[0]))


class TestMakeNormalized:

    def test_normalized_data_spans_zero_to_one(self, y):
        t = AffineTransform.make_normalized(y)
        assert t(y).min() == pytest.approx(0.0)
        assert t(y).max() == pytest.approx(1.0)

    def test_per_column_when_axis_is_given(self, x):
        # each action dimension has its own range, so each needs its own scale
        t = AffineTransform.make_normalized(x, axis=0)
        assert t(x).min(axis=0) == pytest.approx(np.zeros(x.shape[1]))
        assert t(x).max(axis=0) == pytest.approx(np.ones(x.shape[1]))

    def test_inverse_recovers_the_raw_data(self, y):
        t = AffineTransform.make_normalized(y)
        assert t.inv(t(y)) == pytest.approx(y)

    def test_constant_data_maps_to_zero(self):
        # a zero range would divide by zero; the data is shifted but not scaled
        t = AffineTransform.make_normalized(np.full(4, 7.0))
        assert t(np.full(4, 7.0)) == pytest.approx(np.zeros(4))


class TestMakeNormalizedFromBounds:
    """The same [0, 1] map, but from a declared range rather than from the data
    that happens to have been measured."""

    def test_the_bounds_map_onto_zero_and_one(self):
        t = AffineTransform.make_normalized_from_bounds(2.0, 10.0)
        assert t(2.0) == pytest.approx(0.0)
        assert t(10.0) == pytest.approx(1.0)

    def test_it_matches_make_normalized_on_the_data_s_own_range(self, y):
        # the two constructors agree when the bounds are the data's range, so
        # inferring the bounds from the data changes nothing
        t = AffineTransform.make_normalized_from_bounds(y.min(), y.max())
        assert t(y) == pytest.approx(AffineTransform.make_normalized(y)(y))

    def test_per_dimension_bounds(self, x):
        t = AffineTransform.make_normalized_from_bounds(x.min(axis=0), x.max(axis=0))
        assert t(x).min(axis=0) == pytest.approx(np.zeros(x.shape[1]))
        assert t(x).max(axis=0) == pytest.approx(np.ones(x.shape[1]))

    def test_inverse_recovers_the_raw_data(self, y):
        t = AffineTransform.make_normalized_from_bounds(0.0, 20.0)
        assert t.inv(t(y)) == pytest.approx(y)

    def test_data_outside_the_bounds_is_not_clipped(self):
        # the transform is a frame, not a gate; Objective is what rejects an
        # action outside its declared box
        t = AffineTransform.make_normalized_from_bounds(2.0, 10.0)
        assert t(14.0) == pytest.approx(1.5)

    def test_zero_range_bounds_map_to_zero(self):
        t = AffineTransform.make_normalized_from_bounds(7.0, 7.0)
        assert t(7.0) == pytest.approx(0.0)


# ---- Objective -------------------------------------------------------------

class TestObjectiveConstruction:

    def test_standard_y_has_zero_mean_and_unit_std(self, objective):
        assert objective.standard_y.mean() == pytest.approx(0.0)
        assert objective.standard_y.std() == pytest.approx(1.0)

    def test_maximized_objective_keeps_its_ordering(self, x, y):
        obj = Objective('comfort', maximize=True, ydata=y, xdata=x)
        assert obj.standard_y == pytest.approx((y - y.mean()) / y.std())

    def test_minimized_objective_is_flipped(self, x, y):
        # every objective reads larger-is-better once standardized, so the GPs
        # and the Pareto machinery downstream never need to know the direction
        obj = Objective('cost', maximize=False, ydata=y, xdata=x)
        assert obj.standard_y == pytest.approx(-(y - y.mean()) / y.std())

    def test_ytransform_returns_raw_units(self, objective, y):
        # the sign lives inside the transform, so inv() undoes the
        # standardization and the flip -- this is how a GP mean gets plotted in
        # the units the measurement was taken in
        assert objective.ytransform.inv(objective.standard_y) == pytest.approx(y)

    def test_raw_data_is_left_alone(self, objective, y, x):
        assert objective.ydata == pytest.approx(y)
        assert objective.xdata == pytest.approx(x)

    def test_one_dimensional_actions_become_a_column(self, y):
        obj = Objective('cost', False, y, np.arange(5.0))
        assert obj.xdata.shape == (5, 1)


class TestObjectiveActionFrame:
    """An Objective normalizes its own actions onto [0, 1]^K, so it can be
    modelled on its own without a DecoupledObjectives frame around it."""

    def test_normalized_actions_span_zero_to_one(self, objective, x):
        assert objective.normalized_x.min(axis=0) == pytest.approx(np.zeros(x.shape[1]))
        assert objective.normalized_x.max(axis=0) == pytest.approx(np.ones(x.shape[1]))

    def test_each_dimension_gets_its_own_scale(self, objective, x):
        # the column ranges here are 4, 8 and 0.4, so one shared scale could not
        # put all three on [0, 1] -- which is what makes one set of GP
        # lengthscales meaningful across dimensions
        assert objective.normalized_x == pytest.approx(
            (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        )

    def test_xtransform_inverts_back_to_the_raw_actions(self, objective, x):
        # how a GP optimum found in the unit box is reported in raw units
        assert objective.xtransform.inv(objective.normalized_x) == pytest.approx(x)

    def test_the_frame_is_the_objectives_own_not_a_shared_one(self, ragged, x):
        # comfort saw only the first three actions. On its own it stretches them
        # to fill [0, 1]; against the shared DecoupledObjectives frame the same
        # actions sit in the lower half
        assert ragged['comfort'].normalized_x.max(axis=0) == pytest.approx(np.ones(3))
        assert ragged.actions('comfort').max(axis=0) == pytest.approx([0.5, 0.5, 0.5])

    def test_added_points_widen_the_frame(self, objective):
        objective.add_points(np.array([8.0, 16.0, 0.9]), np.array([12.0]))
        # the new action is the largest in every dimension, so it is the one
        # that now maps to 1
        assert objective.normalized_x[-1] == pytest.approx(np.ones(3))

    def test_one_dimensional_actions_normalize_as_a_column(self, y):
        obj = Objective('cost', False, y, np.arange(5.0))
        assert obj.normalized_x.shape == (5, 1)
        assert obj.normalized_x[:, 0] == pytest.approx(np.linspace(0.0, 1.0, 5))


class TestObjectiveToRaw:
    """The inverse of standard_y: standardized, larger-is-better values back in
    the units the objective was measured in. The single-objective counterpart of
    DecoupledObjectives.to_raw, so no selector is needed."""

    def test_means_round_trip_to_the_raw_measurements(self, objective, y):
        assert objective.to_raw(objective.standard_y) == pytest.approx(y)

    def test_a_spread_is_scaled_but_not_shifted(self, objective, y):
        # one standardized unit is one raw standard deviation of this objective
        _, got = objective.to_raw(objective.standard_y, np.ones_like(y))
        assert got == pytest.approx(np.full(y.size, y.std()))

    def test_a_minimized_objectives_spread_stays_positive(self, objective, y):
        # the ytransform carries a negative scale here, so inv() would report a
        # negative standard deviation
        _, got = objective.to_raw(objective.standard_y, np.ones_like(y))
        assert np.all(got > 0)

    def test_the_std_is_optional(self, objective):
        assert isinstance(objective.to_raw(objective.standard_y), np.ndarray)

    def test_it_agrees_with_the_collection_version(self, pair, y):
        # one column through DecoupledObjectives.to_raw is the same conversion
        mu = pair.feedback('cost')[:, None]
        assert pair['cost'].to_raw(mu[:, 0]) == pytest.approx(
            pair.to_raw(mu, objs='cost')[:, 0]
        )


class TestObjectiveSign:

    def test_sign_is_positive_when_maximizing(self, x, y):
        assert Objective('comfort', True, y, x).sign == 1.0

    def test_sign_is_negative_when_minimizing(self, x, y):
        assert Objective('cost', False, y, x).sign == -1.0


class TestObjectiveBestAction:

    def test_maximized_objective_picks_the_largest_value(self, x, y):
        obj = Objective('comfort', maximize=True, ydata=y, xdata=x)
        assert obj.best_action() == pytest.approx(x[y.argmax()])

    def test_minimized_objective_picks_the_smallest_value(self, x, y):
        obj = Objective('cost', maximize=False, ydata=y, xdata=x)
        assert obj.best_action() == pytest.approx(x[y.argmin()])


class TestObjectiveAddPoints:

    def test_a_new_point_is_appended(self, objective, y):
        objective.add_points(np.array([5.0, 10.0, 0.6]), np.array([12.0]))
        assert objective.ydata == pytest.approx(np.append(y, 12.0))
        assert objective.xdata[-1] == pytest.approx([5.0, 10.0, 0.6])

    def test_transforms_are_recomputed(self, objective):
        objective.add_points(np.array([5.0, 10.0, 0.6]), np.array([12.0]))
        # the mean moved, so the old transform would no longer give zero mean
        assert objective.standard_y.mean() == pytest.approx(0.0)

    def test_several_points_are_appended_at_once(self, objective, y):
        objective.add_points(np.array([[5.0, 10.0, 0.6], [6.0, 12.0, 0.7]]),
                             np.array([12.0, 14.0]))
        assert objective.ydata == pytest.approx(np.append(y, [12.0, 14.0]))
        assert objective.xdata.shape == (y.size + 2, 3)


class TestObjectiveFromEmpty:
    """An objective declared before any measurement exists, then grown."""

    def test_the_name_and_direction_are_carried(self):
        obj = Objective.from_empty('cost', maximize=False)
        assert obj.name == 'cost'
        assert obj.maximize is False
        assert obj.sign == -1.0
        assert obj.column is None

    def test_it_holds_no_measurements(self):
        obj = Objective.from_empty('cost', maximize=False)
        assert obj.xdata.size == 0
        assert obj.ydata.size == 0

    def test_it_has_no_transforms_yet(self):
        # there is no mean or spread to compute, so __post_init__ stops early
        obj = Objective.from_empty('cost', maximize=False)
        assert not hasattr(obj, 'xtransform')
        assert not hasattr(obj, 'ytransform')
        with pytest.raises(AttributeError):
            obj.standard_y

    def test_the_first_points_set_the_action_dimension(self):
        obj = Objective.from_empty('cost', maximize=False)
        obj.add_points(np.array([5.0, 10.0, 0.6]), np.array([12.0]))
        assert obj.xdata.shape == (1, 3)
        assert obj.ydata == pytest.approx([12.0])

    def test_the_first_points_build_the_transforms(self, x, y):
        obj = Objective.from_empty('cost', maximize=False)
        obj.add_points(x, y)
        assert obj.standard_y.mean() == pytest.approx(0.0)
        assert obj.standard_y.std() == pytest.approx(1.0)
        assert obj.normalized_x.min(axis=0) == pytest.approx(np.zeros(3))
        assert obj.normalized_x.max(axis=0) == pytest.approx(np.ones(3))

    def test_points_can_be_added_one_at_a_time(self, x, y):
        obj = Objective.from_empty('cost', maximize=False)
        for action, value in zip(x, y):
            obj.add_points(action, np.array([value]))

        assert obj.xdata == pytest.approx(x)
        assert obj.ydata == pytest.approx(y)

    def test_growing_from_empty_matches_direct_construction(self, x, y):
        # the two routes to the same five measurements must agree, transforms
        # included, or an objective's history would change its model
        grown = Objective.from_empty('cost', maximize=False)
        grown.add_points(x, y)
        direct = Objective('cost', maximize=False, ydata=y.copy(), xdata=x.copy())
        assert grown.xdata == pytest.approx(direct.xdata)
        assert grown.ydata == pytest.approx(direct.ydata)
        assert grown.standard_y == pytest.approx(direct.standard_y)
        assert grown.normalized_x == pytest.approx(direct.normalized_x)

    def test_direction_still_holds_once_data_arrives(self, x, y):
        # standard_y is larger-is-better, so a minimized objective's largest
        # standardized value sits at its smallest measurement
        obj = Objective.from_empty('cost', maximize=False)
        obj.add_points(x, y)
        assert obj.best_action() == pytest.approx(x[y.argmin()])


class TestObjectiveActionBounds:
    """Declared action bounds pin the [0, 1]^K frame, so it does not move as
    measurements arrive -- which is what makes a lengthscale, or a floor under
    one, mean the same thing at every step of a sequential run."""

    def test_the_frame_exists_before_any_measurement(self):
        obj = Objective.from_empty('cost', maximize=False,
                                   action_bounds=(-5.0, 5.0))
        assert obj.xtransform(np.array([-5.0, 0.0, 5.0])) == pytest.approx([0.0, 0.5, 1.0])

    def test_the_frame_does_not_move_as_points_arrive(self):
        # without bounds the first point would sit at 0 and be pushed to 1 by
        # the second, so a model fit after each step would see a different box
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0))
        obj.add_points(np.array([[2.0]]), np.array([1.0]))
        assert obj.normalized_x == pytest.approx(np.array([[0.2]]))

        obj.add_points(np.array([[8.0]]), np.array([3.0]))
        assert obj.normalized_x == pytest.approx(np.array([[0.2], [0.8]]))

    def test_without_bounds_the_frame_follows_the_data(self, x):
        # the control for the test above: this is the default behaviour
        obj = Objective.from_empty('cost', maximize=False)
        obj.add_points(x[:2], np.array([1.0, 3.0]))
        assert obj.normalized_x[0] == pytest.approx(np.zeros(3))

    def test_bounds_may_be_given_per_dimension(self, x, y):
        obj = Objective('cost', False, y.copy(), x.copy(),
                        action_bounds=(np.zeros(3), np.array([8.0, 16.0, 1.0])))
        # every column reaches half of its own declared range
        assert obj.normalized_x.max(axis=0) == pytest.approx([0.5, 0.5, 0.5])

    def test_the_inverse_still_recovers_raw_actions(self, x, y):
        obj = Objective('cost', False, y.copy(), x.copy(),
                        action_bounds=(np.zeros(3), np.array([8.0, 16.0, 1.0])))
        assert obj.xtransform.inv(obj.normalized_x) == pytest.approx(x)

    def test_action_box_reports_the_bounds_when_pinned(self, x, y):
        pinned = Objective('cost', False, y.copy(), x.copy(),
                           action_bounds=(0.0, 20.0))
        assert pinned.action_box() == (0.0, 20.0)

    def test_action_box_falls_back_to_the_measured_range(self, objective, x):
        low, high = objective.action_box()
        assert low == pytest.approx(x.min(axis=0))
        assert high == pytest.approx(x.max(axis=0))

    def test_an_empty_objective_without_bounds_has_no_box(self):
        assert Objective.from_empty('cost', maximize=False).action_box() is None


class TestObjectiveActionsMustLieInTheBox:
    """Declaring bounds is a claim about where the experiment lives, so an
    action outside them is a bug -- a wrong unit, a swapped pair, a design from
    another run -- rather than a point to normalize past 1."""

    def test_an_action_outside_the_box_is_rejected_on_construction(self, x, y):
        with pytest.raises(ValueError):
            Objective('cost', False, y, x, action_bounds=(0.0, 1.0))

    def test_an_action_outside_the_box_is_rejected_on_add_points(self):
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0))
        with pytest.raises(ValueError):
            obj.add_points(np.array([[15.0]]), np.array([1.0]))

    def test_below_the_box_is_rejected_too(self):
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0))
        with pytest.raises(ValueError):
            obj.add_points(np.array([[-0.5]]), np.array([1.0]))

    def test_one_bad_dimension_is_enough(self, x, y):
        # the third column runs to 0.5 and is declared to stop at 0.2
        with pytest.raises(ValueError):
            Objective('cost', False, y, x,
                      action_bounds=(np.zeros(3), np.array([8.0, 16.0, 0.2])))

    def test_the_error_names_the_offending_measurements(self, x, y):
        with pytest.raises(ValueError, match=r'\[3, 4\]'):
            Objective('cost', False, y, x,
                      action_bounds=(np.zeros(3), np.array([8.0, 16.0, 0.35])))

    def test_the_boundary_itself_is_accepted(self):
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0))
        obj.add_points(np.array([[0.0], [10.0]]), np.array([1.0, 3.0]))
        assert obj.normalized_x == pytest.approx(np.array([[0.0], [1.0]]))

    def test_float_round_off_outside_the_boundary_is_accepted(self):
        # an action optimized onto the boundary comes back a hair outside it,
        # and losing a measurement to the last bit of a double is not a bug
        # worth reporting
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(-5.0, 5.0))
        obj.add_points(np.array([[np.nextafter(5.0, np.inf)]]), np.array([1.0]))
        assert obj.normalized_x == pytest.approx(np.array([[1.0]]))

    def test_the_slack_does_not_cover_a_real_overrun(self):
        # the tolerance is 1e-9 of the span, so a millimetre past a 10-unit box
        # still raises
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(-5.0, 5.0))
        with pytest.raises(ValueError):
            obj.add_points(np.array([[5.001]]), np.array([1.0]))

    def test_a_rejected_point_is_not_left_in_the_record(self):
        # add_points appends before __post_init__ validates, so the append has
        # to be undone -- a caller that catches the error must not be handed a
        # objective holding the point it just refused
        obj = Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0))
        obj.add_points(np.array([[2.0]]), np.array([1.0]))
        with pytest.raises(ValueError):
            obj.add_points(np.array([[15.0]]), np.array([3.0]))

        assert obj.xdata == pytest.approx(np.array([[2.0]]))
        assert obj.ydata == pytest.approx([1.0])
        assert obj.normalized_x == pytest.approx(np.array([[0.2]]))

    def test_unbounded_objectives_accept_anything(self, x, y):
        # the check exists only where a box was declared
        Objective('cost', False, y, x * 1e6)


class TestObjectiveFromDataFrame:

    @pytest.fixture
    def df(self, y, x):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({
            'Cost': y, 'a0': x[:, 0], 'a1': x[:, 1], 'a2': x[:, 2],
        })

    def test_columns_are_pulled_out_as_arrays(self, df, y, x):
        obj = Objective.from_df(df, 'Cost', ['a0', 'a1', 'a2'], maximize=False)
        assert isinstance(obj.ydata, np.ndarray)
        assert isinstance(obj.xdata, np.ndarray)
        assert obj.ydata == pytest.approx(y)
        assert obj.xdata == pytest.approx(x)

    def test_name_defaults_to_the_column(self, df):
        obj = Objective.from_df(df, 'Cost', ['a0', 'a1', 'a2'], maximize=False)
        assert obj.name == 'Cost'

    def test_name_overrides_the_column(self, df):
        obj = Objective.from_df(df, 'Cost', ['a0', 'a1', 'a2'], False, name='Metabolic Cost')
        assert obj.name == 'Metabolic Cost'

    def test_source_column_is_kept_when_the_name_differs(self, df):
        obj = Objective.from_df(df, 'Cost', ['a0', 'a1', 'a2'], False, name='Metabolic Cost')
        assert obj.column == 'Cost'


class TestObjectiveFromSynthetic:
    """`bounds` is the subtle part: BoTorch wants one (low, high) pair per
    dimension, rejects actions outside them, and requires custom bounds to
    contain a known optimizer. Spanning both the data and the function's own
    default box satisfies all three."""

    @staticmethod
    def truth(function, actions):
        """The function evaluated directly, as the reference for the noiseless case."""
        with torch.no_grad():
            return function(dim=np.atleast_2d(actions).shape[1])(
                torch.as_tensor(actions, dtype=torch.float64), noise=False
            ).numpy()

    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_every_action_dimension_is_supported(self, dim):
        actions = np.linspace(-4.0, 4.0, 6 * dim).reshape(6, dim)
        obj = Objective.from_synthetic(Levy, actions, maximize=False,
                                       rel_noise_std=0.1, seed=0)
        assert obj.xdata.shape == (6, dim)
        assert obj.ydata.shape == (6,)
        assert np.all(np.isfinite(obj.ydata))

    def test_one_dimensional_actions_gain_a_column(self):
        obj = Objective.from_synthetic(Levy, np.linspace(-5.0, 5.0, 7),
                                       maximize=False, rel_noise_std=0.0, seed=0)
        assert obj.xdata.shape == (7, 1)

    def test_actions_are_kept_as_given(self):
        actions = np.linspace(-4.0, 4.0, 10).reshape(5, 2)
        obj = Objective.from_synthetic(Levy, actions, maximize=False,
                                       rel_noise_std=0.3, seed=0)
        assert obj.xdata == pytest.approx(actions)

    def test_zero_noise_reproduces_the_function(self):
        actions = np.linspace(-4.0, 4.0, 10).reshape(5, 2)
        obj = Objective.from_synthetic(Levy, actions, maximize=False,
                                       rel_noise_std=0.0, seed=0)
        assert obj.ydata == pytest.approx(self.truth(Levy, actions))

    def test_integer_actions_keep_full_precision(self):
        # float32 evaluation would agree only to ~1e-7
        actions = np.array([[-4, -2], [0, 1], [2, 3], [4, 4]])
        obj = Objective.from_synthetic(Levy, actions, maximize=False,
                                       rel_noise_std=0.0, seed=0)
        assert obj.ydata == pytest.approx(self.truth(Levy, actions.astype(float)),
                                          rel=1e-12)

    def test_the_same_seed_gives_the_same_measurements(self):
        actions = np.linspace(-4.0, 4.0, 12).reshape(6, 2)
        kwargs = dict(maximize=False, rel_noise_std=0.2)
        a = Objective.from_synthetic(Levy, actions, seed=7, **kwargs)
        b = Objective.from_synthetic(Levy, actions, seed=7, **kwargs)
        assert a.ydata == pytest.approx(b.ydata)

    def test_a_different_seed_gives_different_measurements(self):
        actions = np.linspace(-4.0, 4.0, 12).reshape(6, 2)
        kwargs = dict(maximize=False, rel_noise_std=0.2)
        a = Objective.from_synthetic(Levy, actions, seed=7, **kwargs)
        b = Objective.from_synthetic(Levy, actions, seed=8, **kwargs)
        assert not np.allclose(a.ydata, b.ydata)

    def test_noise_is_a_fraction_of_the_truths_spread(self):
        actions = np.linspace(-4.0, 4.0, 200).reshape(100, 2)
        truth = self.truth(Levy, actions)
        residuals = [
            Objective.from_synthetic(Levy, actions, maximize=False,
                                     rel_noise_std=rel, seed=3).ydata - truth
            for rel in (0.1, 0.4)
        ]
        # the same draw scaled by rel, so the ratio of spreads is the ratio of rels
        assert residuals[1].std() / residuals[0].std() == pytest.approx(4.0, rel=1e-9)

        # exactly rel * spread * the draw that seed 3 produces
        draw = np.random.default_rng(3).standard_normal(truth.shape)
        assert residuals[0] == pytest.approx(0.1 * truth.std() * draw, rel=1e-9)

    def test_maximize_sets_the_direction(self):
        actions = np.linspace(-4.0, 4.0, 10).reshape(5, 2)
        kwargs = dict(rel_noise_std=0.0, seed=0)
        lo = Objective.from_synthetic(Levy, actions, maximize=False, **kwargs)
        hi = Objective.from_synthetic(Levy, actions, maximize=True, **kwargs)

        assert (lo.sign, hi.sign) == (-1.0, 1.0)
        assert lo.ydata == pytest.approx(hi.ydata)          # same measurements
        assert lo.standard_y == pytest.approx(-hi.standard_y)
        # standard_y is larger-is-better, so minimizing peaks at the smallest value
        assert np.argmax(lo.standard_y) == np.argmin(lo.ydata)

    def test_a_design_that_misses_the_optimum_still_builds(self):
        # Levy's optimum is at (1, ..., 1), strictly outside this box
        actions = np.linspace(3.0, 5.0, 10).reshape(5, 2)
        obj = Objective.from_synthetic(Levy, actions, maximize=False,
                                       rel_noise_std=0.1, seed=0)
        assert np.all(np.isfinite(obj.ydata))

    def test_actions_outside_the_default_box_still_build(self):
        # StyblinskiTang's default box is [-5, 5]
        actions = np.linspace(-8.0, 8.0, 10).reshape(5, 2)
        obj = Objective.from_synthetic(StyblinskiTang, actions, maximize=False,
                                       rel_noise_std=0.0, seed=0)
        assert np.all(np.isfinite(obj.ydata))

    def test_name_is_passed_through(self):
        obj = Objective.from_synthetic(Levy, np.linspace(-4.0, 4.0, 6).reshape(3, 2),
                                       maximize=False, rel_noise_std=0.0, seed=0,
                                       name='Levy cost')
        assert obj.name == 'Levy cost'
        assert obj.column is None


# ---- DecoupledObjectives ---------------------------------------------------

class TestDecoupledObjectivesBasics:

    def test_names_are_listed_in_order(self, pair):
        assert pair.names == ['cost', 'comfort']

    def test_num_objectives_counts_the_objectives(self, pair):
        assert pair.num_objectives == 2
        assert len(pair) == 2

    def test_columns_are_none_when_not_read_from_a_dataframe(self, pair):
        assert pair.columns == [None, None]


class TestDecoupledObjectivesSelection:
    """One selector shape is shared by __getitem__, feedback, actions, min,
    max, range and add_point: a name, a position, a list of either, a slice,
    or None for all of them."""

    def test_select_by_name(self, pair):
        assert pair['comfort'].maximize is True

    def test_select_by_position(self, pair):
        assert pair[1].name == 'comfort'

    def test_select_by_list_of_names(self, pair):
        assert pair[['comfort', 'cost']].names == ['comfort', 'cost']

    def test_select_by_list_of_positions(self, pair):
        assert pair[[1, 0]].names == ['comfort', 'cost']

    def test_slice_gives_another_collection(self, pair):
        sub = pair[:1]
        assert isinstance(sub, DecoupledObjectives)
        assert sub.names == ['cost']

    def test_a_single_selector_gives_a_bare_objective(self, pair):
        assert isinstance(pair['cost'], Objective)
        assert isinstance(pair[0], Objective)

    def test_unknown_names_are_rejected(self, pair):
        with pytest.raises(ValueError):
            pair['speed']

    def test_other_selector_types_are_rejected(self, pair):
        with pytest.raises(TypeError):
            pair[1.5]


class TestDecoupledObjectivesReadout:

    def test_feedback_is_the_standardized_values(self, pair, y):
        assert pair.feedback('cost') == pytest.approx(-(y - y.mean()) / y.std())

    def test_feedback_for_several_objectives_is_a_list(self, pair):
        got = pair.feedback(['cost', 'comfort'])
        assert isinstance(got, list) and len(got) == 2

    def test_feedback_defaults_to_every_objective(self, pair):
        assert len(pair.feedback()) == 2

    def test_actions_are_normalized(self, pair):
        got = pair.actions('cost')
        assert got.min(axis=0) == pytest.approx(np.zeros(3))
        assert got.max(axis=0) == pytest.approx(np.ones(3))

    def test_feedback_and_actions_line_up_row_by_row(self, ragged):
        # what a per-objective GP is fit on: three (action, value) pairs for
        # comfort, five for cost
        for values, actions in zip(ragged.feedback(), ragged.actions()):
            assert values.shape[0] == actions.shape[0]


class TestDecoupledObjectivesToRaw:
    """The inverse of feedback(): standardized, larger-is-better columns back in
    the units each objective was measured in."""

    @pytest.fixture
    def mu(self, pair):
        return np.column_stack(pair.feedback())

    def test_means_round_trip_to_the_raw_measurements(self, pair, mu, y):
        assert pair.to_raw(mu) == pytest.approx(np.column_stack([y, 2.0 * y]))

    def test_a_spread_is_scaled_but_not_shifted(self, pair, mu, y):
        # one standardized unit is one raw standard deviation of that objective,
        # and comfort is twice cost, so its standard deviation is twice as wide
        _, got = pair.to_raw(mu, np.ones_like(mu))
        assert got == pytest.approx(np.column_stack([
            np.full(len(y), y.std()), np.full(len(y), 2.0 * y.std())
        ]))

    def test_a_minimized_objectives_spread_stays_positive(self, pair, mu):
        # cost's ytransform carries a negative scale, so inv() would report a
        # negative standard deviation here
        _, got = pair.to_raw(mu, np.ones_like(mu))
        assert np.all(got > 0)

    def test_the_std_is_optional(self, pair, mu):
        assert isinstance(pair.to_raw(mu), np.ndarray)

    def test_columns_follow_the_selector(self, pair, y):
        mu = np.column_stack([pair.feedback('comfort'), pair.feedback('cost')])
        assert pair.to_raw(mu, objs=['comfort', 'cost']) == pytest.approx(
            np.column_stack([2.0 * y, y])
        )


class TestDecoupledObjectivesShareOneActionFrame:

    def test_the_frame_spans_every_objectives_actions(self, ragged):
        # cost covers the full range of each action dimension, so it is what
        # sets the box, and it normalizes to [0, 1]
        got = ragged.actions('cost')
        assert got.min(axis=0) == pytest.approx(np.zeros(3))
        assert got.max(axis=0) == pytest.approx(np.ones(3))

    def test_an_objective_over_a_sub_box_maps_into_a_sub_interval(self, ragged, x):
        # comfort only saw the first three actions. Normalizing it on its own
        # range would stretch it to [0, 1] and misplace it relative to cost;
        # against the shared frame it lands where it actually is
        got = ragged.actions('comfort')
        expected = (x[:3] - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        assert got == pytest.approx(expected)
        assert got.max(axis=0) == pytest.approx([0.5, 0.5, 0.5])


class TestDecoupledObjectivesRanges:

    def test_min_and_max_are_per_objective(self, pair, y):
        standardized = (y - y.mean()) / y.std()
        # comfort is twice cost in raw units, but standardizing divides that
        # factor out, so only the flip on the minimized cost is left
        assert pair.min() == pytest.approx([-standardized.max(), standardized.min()])
        assert pair.max() == pytest.approx([-standardized.min(), standardized.max()])

    def test_range_pairs_each_min_with_its_max(self, pair):
        got = pair.range()
        assert got.shape == (2, 2)   # (num_objs, [min, max])
        assert got[:, 0] == pytest.approx(pair.min())
        assert got[:, 1] == pytest.approx(pair.max())

    def test_a_single_objective_gives_one_pair(self, pair):
        assert pair.range('cost').shape == (2,)
        assert pair.range('cost') == pytest.approx(
            [pair['cost'].standard_y.min(), pair['cost'].standard_y.max()]
        )


class TestDecoupledObjectivesGrowth:

    def test_from_empty_starts_with_no_objectives(self):
        empty = DecoupledObjectives.from_empty()
        assert empty.names == []
        assert empty.num_objectives == 0

    def test_an_objective_can_be_added_to_an_empty_collection(self, x, y):
        objs = DecoupledObjectives.from_empty()
        objs.add_objective(Objective('cost', False, y.copy(), x.copy()))
        assert objs.names == ['cost']
        assert objs.actions('cost').max(axis=0) == pytest.approx(np.ones(3))

    def test_add_objective_extends_the_collection(self, pair, x, y):
        pair.add_objective(Objective('speed', True, y.copy(), x.copy()))
        assert pair.names == ['cost', 'comfort', 'speed']

    def test_add_point_only_touches_the_selected_objective(self, pair, y):
        pair.add_point('cost', np.array([5.0, 10.0, 0.6]), np.array([12.0]))
        assert pair['cost'].ydata.size == y.size + 1
        assert pair['comfort'].ydata.size == y.size

    def test_add_point_can_touch_several_objectives(self, pair, y):
        pair.add_point(['cost', 'comfort'], np.array([5.0, 10.0, 0.6]), np.array([12.0]))
        assert pair['cost'].ydata.size == y.size + 1
        assert pair['comfort'].ydata.size == y.size + 1

    def test_add_point_widens_the_shared_frame(self, pair):
        # the new action sits outside the old box, so the frame has to grow
        pair.add_point('cost', np.array([8.0, 16.0, 0.9]), np.array([12.0]))
        assert pair.actions('comfort').max(axis=0) == pytest.approx([0.5, 0.5, 0.5])


class TestDecoupledObjectivesSharedFrame:
    """The shared frame spans every objective's box: its action_bounds where
    they are pinned and its measured actions where they are not."""

    def test_a_pinned_objective_sets_the_frame_for_all_of_them(self, x, y):
        # 'cost' declares twice the range it has measured, and 'comfort', which
        # declares nothing, is normalized into that same wider box
        objs = DecoupledObjectives([
            Objective('cost', False, y.copy(), x.copy(),
                      action_bounds=(np.zeros(3), np.array([8.0, 16.0, 1.0]))),
            Objective('comfort', True, y.copy(), x.copy()),
        ])
        assert objs.actions('comfort').max(axis=0) == pytest.approx([0.5, 0.5, 0.5])

    def test_the_frame_is_the_union_not_the_last_word(self, x, y):
        # 'cost' declares a box narrower than 'comfort' has measured, and keeps
        # inside it; the shared frame still has to contain everything either
        # objective can produce
        objs = DecoupledObjectives([
            Objective('cost', False, y.copy(), x.copy() / 10, action_bounds=(0.0, 1.0)),
            Objective('comfort', True, y.copy(), x.copy()),
        ])
        # low is min(0, x.min) and high is max(1, x.max), per dimension
        assert objs.actions('comfort').min(axis=0) == pytest.approx([0.0, 0.0, 0.1])
        assert objs.actions('comfort').max(axis=0) == pytest.approx([1.0, 1.0, 0.5])

    def test_a_pinned_frame_does_not_move_as_points_arrive(self, x, y):
        objs = DecoupledObjectives([
            Objective.from_empty('cost', maximize=False, action_bounds=(0.0, 10.0)),
        ])
        objs.add_point('cost', np.array([2.0]), np.array([1.0]))
        assert objs.actions('cost') == pytest.approx(np.array([[0.2]]))

        objs.add_point('cost', np.array([8.0]), np.array([3.0]))
        assert objs.actions('cost') == pytest.approx(np.array([[0.2], [0.8]]))

    def test_objectives_with_neither_bounds_nor_data_have_no_frame(self):
        objs = DecoupledObjectives([Objective.from_empty('cost', maximize=False)])
        assert objs.xtransform is None

    def test_it_matches_the_data_derived_frame_when_nothing_is_pinned(self, ragged, x):
        # the union of the measured ranges is the box make_normalized would have
        # built over the concatenated actions
        assert ragged.actions('cost').min(axis=0) == pytest.approx(np.zeros(3))
        assert ragged.actions('cost').max(axis=0) == pytest.approx(np.ones(3))


class TestDecoupledObjectivesAreDecoupled:

    def test_objectives_may_hold_different_numbers_of_points(self, ragged, y):
        # the point of "decoupled": one objective can be measured more often
        # than another, e.g. a metabolic cost measured every trial against a
        # comfort survey answered occasionally
        assert ragged.num_objectives == 2
        assert ragged.feedback('cost').size == y.size
        assert ragged.feedback('comfort').size == 3


class TestDecoupledObjectivesFromDataFrame:

    @pytest.fixture
    def df(self, y, x):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({
            'Cost': y, 'Speed': 2.0 * y,
            'a0': x[:, 0], 'a1': x[:, 1], 'a2': x[:, 2],
        })

    def test_one_objective_per_column(self, df):
        objs = DecoupledObjectives.from_df(
            df, ['Cost', 'Speed'], ['a0', 'a1', 'a2'], maximize=False
        )
        assert objs.names == ['Cost', 'Speed']

    def test_a_single_maximize_flag_applies_to_all(self, df):
        objs = DecoupledObjectives.from_df(
            df, ['Cost', 'Speed'], ['a0', 'a1', 'a2'], maximize=False
        )
        assert [o.maximize for o in objs.objectives] == [False, False]

    def test_maximize_may_be_given_per_objective(self, df):
        objs = DecoupledObjectives.from_df(
            df, ['Cost', 'Speed'], ['a0', 'a1', 'a2'], maximize=[False, True]
        )
        assert [o.maximize for o in objs.objectives] == [False, True]

    def test_names_override_the_columns(self, df):
        objs = DecoupledObjectives.from_df(
            df, ['Cost', 'Speed'], ['a0', 'a1', 'a2'], False,
            names=['Metabolic Cost', '10m walk time']
        )
        assert objs.names == ['Metabolic Cost', '10m walk time']

    def test_columns_property_reports_the_source_columns(self, df):
        objs = DecoupledObjectives.from_df(
            df, ['Cost', 'Speed'], ['a0', 'a1', 'a2'], False,
            names=['Metabolic Cost', '10m walk time']
        )
        assert objs.columns == ['Cost', 'Speed']
