"""Tests for AffineTransform, Objective and DecoupledObjectives.

Each test states one property in the terms the class name promises: an affine
transform is invertible, a "centered" transform removes the mean, a
"normalized" transform maps data onto [0, 1], and a set of *decoupled*
objectives lets each objective carry its own measurements, in its own number,
over one shared action frame.
"""

import numpy as np
import pytest

from pypolar.optimization.objectives import (
    AffineTransform,
    DecoupledObjectives,
    Objective,
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
