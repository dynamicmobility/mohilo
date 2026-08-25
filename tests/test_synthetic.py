"""Tests for the synthetic groundtruths.

The module's contract is that a noise stated as a fraction of a function's own
spread means the same difficulty across functions whose ranges differ by orders
of magnitude, and that a multi-objective truth is m of those, independent. The
tests below pin the spread measurement, the noise stream, the scalarization
`MO2SO` performs, and which instances `construct_function` will and will not
build.
"""

import numpy as np
import pytest

from botorch.test_functions import multi_objective, synthetic

from pypolar.feedback.synthetic import (
    MO_SYNTHETIC_FUNCTIONS,
    SYNTHETIC_1D_FUNCTIONS,
    SYNTHETIC_FUNCTIONS,
    MO2SO,
    MOSyntheticOracle,
    SyntheticOracle,
    construct_function,
    truth_at,
)

BOX  = 5.0
SEED = 3

# a Sobol scan of the whole box, so a spread read off it is the module's own
N_SPREAD = 256

# above 1D Levy's largest value on [-5, 5] (3.884), so the whole front counts
LEVY_REF = 4.0


@pytest.fixture
def levy():
    """A 1D Levy instance on [-5, 5]."""
    return construct_function(synthetic.Levy, dim=1, box=BOX, seed=SEED)


@pytest.fixture
def branin_currin():
    """A 2-objective, 2D multi-objective instance."""
    return multi_objective.BraninCurrin()


class TestConstructFunction:

    def test_the_requested_box_is_the_one_it_gets(self, levy):
        np.testing.assert_allclose(levy.bounds.numpy(), [[-BOX], [BOX]])
        assert levy.dim == 1

    def test_a_constant_instance_is_refused(self):
        # Rosenbrock sums over range(dim - 1), so at dim 1 it is identically zero
        assert construct_function(synthetic.Rosenbrock, dim=1, box=BOX) is None

    def test_the_1d_registry_is_what_it_measures(self):
        built = {name for name, func in SYNTHETIC_FUNCTIONS.items()
                 if construct_function(func, dim=1, box=BOX) is not None}
        assert built == set(SYNTHETIC_1D_FUNCTIONS)

    def test_a_refused_box_falls_back_to_the_functions_own_bounds(self):
        # StyblinskiTang's optimizer is at -2.904, so botorch refuses a box of
        # half-width 1 as holding none; the instance is built on [-5, 5] instead
        truth = construct_function(synthetic.StyblinskiTang, dim=1, box=1.0)

        assert truth is not None and truth.dim == 1
        np.testing.assert_allclose(truth.bounds.numpy(), [[-5.0], [5.0]])

    def test_the_1d_registry_does_not_depend_on_the_box(self):
        for box in (1.0, 2.0, 5.0):
            built = {name for name, func in SYNTHETIC_FUNCTIONS.items()
                     if construct_function(func, dim=1, box=box) is not None}
            assert built == set(SYNTHETIC_1D_FUNCTIONS), box

    def test_a_multi_objective_problem_keeps_its_own_bounds(self, branin_currin):
        truth = construct_function(multi_objective.BraninCurrin, dim=2, box=BOX)

        # no multi-objective problem takes a `bounds` argument, so the box is inert
        np.testing.assert_allclose(truth.bounds.numpy(),
                                   branin_currin.bounds.numpy())

    def test_num_objectives_reaches_the_families_that_take_one(self):
        truth = construct_function(multi_objective.DTLZ2, dim=4, box=BOX,
                                   num_objectives=3)
        assert truth.num_objectives == 3 and truth.dim == 4


class TestSyntheticOracle:

    def test_the_noiseless_call_is_the_truth(self, levy):
        sf = SyntheticOracle(levy, rel_noise_std=0.5, n_spread=N_SPREAD, seed=SEED)
        X  = np.linspace(-BOX, BOX, 7)[:, None]

        np.testing.assert_allclose(sf(X, noise=False), truth_at(levy, X))

    def test_range_and_std_are_the_scans_own(self, levy):
        X = np.linspace(-BOX, BOX, 5)[:, None]
        rng = SyntheticOracle(levy, measure='range', n_spread=N_SPREAD, seed=SEED)
        std = SyntheticOracle(levy, measure='std', n_spread=N_SPREAD, seed=SEED)

        assert rng.measure_spread == rng.ptp == rng.sample_max - rng.sample_min
        assert 0 < std.measure_spread < rng.measure_spread

        # the spread is a property of the scan, not of what is measured after it
        np.testing.assert_allclose(rng(X, noise=False), std(X, noise=False))

    def test_the_noise_is_that_fraction_of_the_spread(self, levy):
        sf = SyntheticOracle(levy, rel_noise_std=0.25, n_spread=N_SPREAD, seed=SEED)
        assert sf.noise_std == pytest.approx(0.25 * sf.measure_spread)

    def test_zero_relative_noise_is_noiseless(self, levy):
        sf = SyntheticOracle(levy, rel_noise_std=0.0, n_spread=N_SPREAD, seed=SEED)
        X  = np.linspace(-BOX, BOX, 7)[:, None]

        np.testing.assert_allclose(sf(X), sf(X, noise=False))

    def test_repeated_calls_advance_one_stream(self, levy):
        sf = SyntheticOracle(levy, rel_noise_std=0.5, n_spread=N_SPREAD, seed=SEED)
        X  = np.linspace(-BOX, BOX, 7)[:, None]

        assert not np.allclose(sf(X), sf(X))

    def test_the_same_seed_replays_the_same_stream(self, levy):
        X = np.linspace(-BOX, BOX, 7)[:, None]
        kwargs = dict(rel_noise_std=0.5, n_spread=N_SPREAD, seed=SEED)

        np.testing.assert_allclose(SyntheticOracle(levy, **kwargs)(X),
                                   SyntheticOracle(levy, **kwargs)(X))

    def test_an_unknown_measure_is_refused(self, levy):
        with pytest.raises(ValueError, match='measure'):
            SyntheticOracle(levy, measure='variance', n_spread=N_SPREAD)


class TestFromName:

    def test_it_builds_the_named_function(self, levy):
        sf = SyntheticOracle.from_name(func='Levy', dim=1, box=BOX, seed=SEED,
                                       n_spread=N_SPREAD)
        X = np.linspace(-BOX, BOX, 7)[:, None]

        np.testing.assert_allclose(sf(X, noise=False), truth_at(levy, X))

    def test_the_same_arguments_give_the_same_oracle(self):
        spec = dict(func='Levy', dim=1, box=BOX, seed=SEED, rel_noise_std=0.1)
        assert (SyntheticOracle.from_name(**spec).noise_std
                == SyntheticOracle.from_name(**spec).noise_std)

    def test_a_function_with_no_instance_at_that_dim_raises(self):
        with pytest.raises(ValueError, match='Rosenbrock'):
            SyntheticOracle.from_name(func='Rosenbrock', dim=1, box=BOX)


class TestMO2SO:

    def test_a_unit_weight_selects_one_column(self, branin_currin):
        X = np.random.default_rng(SEED).random((6, 2))
        Y = truth_at(branin_currin, X)

        for j, w in enumerate(np.eye(branin_currin.num_objectives)):
            np.testing.assert_allclose(truth_at(MO2SO(branin_currin, w), X), Y[:, j])

    def test_a_mixed_weight_is_the_weighted_sum(self, branin_currin):
        X = np.random.default_rng(SEED).random((6, 2))
        w = np.array([0.25, 0.75])

        np.testing.assert_allclose(truth_at(MO2SO(branin_currin, w), X),
                                   truth_at(branin_currin, X) @ w)

    def test_it_carries_the_bounds_and_dim_a_truth_is_asked_for(self, branin_currin):
        scalar = MO2SO(branin_currin, np.array([1.0, 0.0]))

        assert scalar.dim == branin_currin.dim
        np.testing.assert_allclose(scalar.bounds.numpy(),
                                   branin_currin.bounds.numpy())


class TestMOSyntheticOracle:

    @pytest.fixture
    def oracle(self, branin_currin):
        return MOSyntheticOracle(branin_currin, rel_noise_std=0.1,
                                 n_spread=N_SPREAD, seed=SEED)

    def test_it_is_one_oracle_per_objective(self, oracle, branin_currin):
        assert len(oracle) == branin_currin.num_objectives
        assert oracle[0] is oracle.objective(0)
        assert oracle.measure_spread.shape == (branin_currin.num_objectives,)

    def test_the_noiseless_call_is_the_truth(self, oracle, branin_currin):
        X = np.random.default_rng(SEED).random((5, 2))

        np.testing.assert_allclose(oracle(X, noise=False), truth_at(branin_currin, X))

    def test_each_column_carries_its_own_objectives_noise(self, oracle):
        assert oracle.noise_std.shape == (2,)
        np.testing.assert_allclose(oracle.noise_std, 0.1 * oracle.measure_spread)

        # the spreads differ by an order of magnitude, which is the whole reason
        # the noise is stated relative to each objective's own
        assert oracle.measure_spread[0] > 10 * oracle.measure_spread[1]

    def test_the_noise_is_independent_across_objectives(self, branin_currin):
        oracle = MOSyntheticOracle(branin_currin, rel_noise_std=0.5,
                                   n_spread=N_SPREAD, seed=SEED)
        X = np.random.default_rng(SEED).random((256, 2))

        residual = oracle(X) - oracle(X, noise=False)
        corr     = np.corrcoef(residual, rowvar=False)[0, 1]

        assert abs(corr) < 0.2

    def test_a_list_of_scalar_truths_is_the_same_object(self, levy):
        # a list of functions carries no reference point, so one is required
        oracle = MOSyntheticOracle([levy, levy], ref_point=LEVY_REF, n_spread=N_SPREAD,
                                   seed=SEED)
        X = np.linspace(-BOX, BOX, 7)[:, None]

        assert len(oracle) == 2
        np.testing.assert_allclose(oracle(X, noise=False)[:, 0], truth_at(levy, X))

    def test_from_name_builds_the_named_problem(self, branin_currin):
        oracle = MOSyntheticOracle.from_name(func='BraninCurrin', dim=2, box=BOX,
                                             seed=SEED, n_spread=N_SPREAD)
        X = np.random.default_rng(SEED).random((5, 2))

        np.testing.assert_allclose(oracle(X, noise=False), truth_at(branin_currin, X))

    def test_from_name_passes_num_objectives_through(self):
        oracle = MOSyntheticOracle.from_name(func='DTLZ2', dim=4, box=BOX,
                                             num_objectives=3, n_spread=N_SPREAD)
        assert len(oracle) == 3


class TestSampledMaxHypervolume:
    """The multi-objective analog of `sample_min`: the best a finite scan of
    the box manages, which is what a hypervolume attained during a run gets
    normalized by."""

    @pytest.mark.parametrize('func,kwargs', [
        ('BraninCurrin', {}),
        ('DTLZ2',        dict(dim=4, num_objectives=2)),
        ('ZDT1',         dict(dim=4, num_objectives=2)),
    ])
    def test_the_scan_finds_most_but_not_all_of_the_true_front(self, func, kwargs):
        """A finite scan lands on a subset of the true front, so its hypervolume
        sits below the analytic maximum botorch carries -- and near it, or
        nothing normalized by it would mean much. The default scan is used
        rather than this module's coarse one, since the ratio is what a coarse
        scan degrades: over six seeds it runs 0.918 to 0.972 at 4096 points and
        drops to 0.70 at 256.
        """
        truth  = getattr(multi_objective, func)(**kwargs)
        oracle = MOSyntheticOracle(truth, seed=SEED)

        assert 0.85 < oracle.sampled_max_hypervolume / truth._max_hv < 1.0

    def test_the_reference_defaults_to_the_truths_own(self, branin_currin):
        oracle = MOSyntheticOracle(branin_currin, n_spread=N_SPREAD, seed=SEED)

        np.testing.assert_allclose(oracle.ref_point, branin_currin.ref_point.numpy())

    def test_a_looser_reference_admits_more_volume(self, branin_currin):
        """The reference is the worst value per objective that still counts, so
        relaxing it can only add volume. This is what pins which way round the
        front is measured from it."""
        tight = MOSyntheticOracle(branin_currin, n_spread=N_SPREAD, seed=SEED)
        loose = MOSyntheticOracle(branin_currin, n_spread=N_SPREAD, seed=SEED,
                                  ref_point=branin_currin.ref_point.numpy() + 10.0)

        assert loose.sampled_max_hypervolume > tight.sampled_max_hypervolume > 0

    def test_a_list_of_functions_requires_a_reference(self, levy):
        # a MultiObjectiveTestProblem carries one; a list of scalar truths does not
        with pytest.raises(ValueError, match='ref_point'):
            MOSyntheticOracle([levy, levy], n_spread=N_SPREAD, seed=SEED)

    def test_one_reference_covers_every_objective(self, levy):
        spec   = dict(n_spread=N_SPREAD, seed=SEED)
        scalar = MOSyntheticOracle([levy, levy], ref_point=LEVY_REF, **spec)
        pair   = MOSyntheticOracle([levy, levy], ref_point=[LEVY_REF] * 2, **spec)

        np.testing.assert_allclose(scalar.ref_point, [LEVY_REF] * 2)
        assert scalar.sampled_max_hypervolume == pair.sampled_max_hypervolume

    def test_the_scan_ignores_the_observation_noise(self, branin_currin):
        """The scan reads the truth with noise=False, so how noisy an oracle is
        cannot move the number its own runs are scored against."""
        spec  = dict(n_spread=N_SPREAD, seed=SEED)
        clean = MOSyntheticOracle(branin_currin, rel_noise_std=0.0, **spec)
        noisy = MOSyntheticOracle(branin_currin, rel_noise_std=0.5, **spec)

        assert clean.sampled_max_hypervolume == noisy.sampled_max_hypervolume


class TestRegistries:

    def test_the_1d_registry_is_a_subset_of_the_full_one(self):
        assert set(SYNTHETIC_1D_FUNCTIONS) <= set(SYNTHETIC_FUNCTIONS)

    def test_every_multi_objective_entry_is_unconstrained(self):
        # a constrained problem's __call__ returns its objectives and leaves the
        # constraints to evaluate_slack, so it would be silently unconstrained here
        for name, func in MO_SYNTHETIC_FUNCTIONS.items():
            assert not hasattr(func, 'evaluate_slack_true'), name
