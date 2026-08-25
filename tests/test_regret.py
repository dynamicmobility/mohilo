"""Tests for the normalized inference regret.

The metric's contract is that it scores a *recommended* action against the
truth's own scanned optimum, in spreads of that truth's range, and that the
result is non-negative whichever direction the truth is optimized in. The tests
below pin the scale against a truth whose regret is known in closed form, pin
the two directions against each other, and pin how far outside [0, 1] the
result can drift when the scan that normalizes it is finite.
"""

import numpy as np
import pytest

from pypolar.feedback.synthetic import SyntheticOracle
from pypolar.optimization.objectives import sample_actions
from pypolar.performance.regret import normalized_inference_regret

BOX  = 5.0
SEED = 3

# points drawn anywhere in the box, to score a recommendation that need not be
# one of the scan's own
N_PROBE = 300

# how far outside [0, 1] a regret may land. The normalizer comes from a finite
# Sobol scan, so `sample_min` sits above the function's analytic minimum and a
# recommendation nearer that minimum scores below zero. Measured across the
# functions below at the default 4096-point scan, the worst drift is 0.032, on
# Ackley, whose optimum is a funnel narrow enough for a scan to miss.
SCAN_SLACK = 0.05

PROBE_FUNCTIONS = [('Levy', 3), ('Ackley', 2), ('Rastrigin', 2), ('DixonPrice', 2)]


class Linear:
    """f(x) = sum(x), exposing the `bounds`, `dim` and `noise` keyword a
    `SyntheticTestFunction` instance does, which is all a `SyntheticOracle`
    asks of a truth.

    Linear is the point: an action a fraction t of the way from the scan's
    argmin to its argmax has a value exactly a fraction t of the way from the
    scan's min to its max, so the regret there is t in closed form, with no
    appeal to what the function returns.
    """

    def __init__(self, dim, box=BOX):
        self.dim    = dim
        self.bounds = np.array([[-box] * dim, [box] * dim], dtype=float)

    def __call__(self, X, noise=True):
        return X.sum(dim=-1)


@pytest.fixture(params=[1, 3])
def linear(request):
    """A linear oracle, at one and at three action dimensions."""
    return SyntheticOracle(Linear(dim=request.param), seed=SEED)


@pytest.fixture
def levy():
    """A 3D Levy oracle on [-5, 5]."""
    return SyntheticOracle.from_name(func='Levy', dim=3, box=BOX, seed=SEED)


def along(oracle, t):
    """The action a fraction t of the way from the scan's argmin to its argmax."""
    return (1 - t) * oracle.sample_argmin + t * oracle.sample_argmax


class TestKnownRegret:

    @pytest.mark.parametrize('t', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_minimizing_scores_the_fraction_travelled_from_the_optimum(self, linear, t):
        assert normalized_inference_regret(along(linear, t), linear) \
            == pytest.approx(t, abs=1e-12)

    @pytest.mark.parametrize('t', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_maximizing_measures_from_the_other_end(self, linear, t):
        assert normalized_inference_regret(along(linear, t), linear, maximize=True) \
            == pytest.approx(1 - t, abs=1e-12)

    def test_the_scanned_extremes_score_zero_and_one(self, levy):
        for maximize, (best, worst) in [(False, (levy.sample_argmin, levy.sample_argmax)),
                                        (True,  (levy.sample_argmax, levy.sample_argmin))]:
            assert normalized_inference_regret(best, levy, maximize=maximize) \
                == pytest.approx(0.0, abs=1e-12)
            assert normalized_inference_regret(worst, levy, maximize=maximize) \
                == pytest.approx(1.0)


    def test_the_normalizer_is_the_range_not_the_noise_measure(self):
        """`measure` picks what the *noise* is a fraction of. A regret is always
        in peak-to-peak spreads, so the two oracles score the same action alike.
        """
        spec = dict(func='Levy', dim=3, box=BOX, seed=SEED)
        rng  = SyntheticOracle.from_name(measure='range', **spec)
        std  = SyntheticOracle.from_name(measure='std', **spec)

        # without this the test passes on any normalizer, since the two coincide
        assert std.measure_spread != std.ptp
        assert normalized_inference_regret(np.full(3, 0.7), std) \
            == pytest.approx(normalized_inference_regret(np.full(3, 0.7), rng))


class TestDirection:

    def test_the_two_directions_sum_to_one(self, levy):
        """A gap measured from the bottom and one measured from the top span the
        whole range between them, whatever the action."""
        X  = sample_actions(bounds=levy.truth.bounds, n=N_PROBE, kind='uniform', seed=SEED)
        lo = np.array([normalized_inference_regret(x, levy) for x in X])
        hi = np.array([normalized_inference_regret(x, levy, maximize=True) for x in X])

        np.testing.assert_allclose(lo + hi, 1.0)

    @pytest.mark.parametrize('func,dim', PROBE_FUNCTIONS)
    @pytest.mark.parametrize('maximize', [False, True])
    def test_regret_stays_within_the_unit_range(self, func, dim, maximize):
        oracle = SyntheticOracle.from_name(func=func, dim=dim, box=BOX, seed=SEED)
        # the analytic optimizers included, since they are what drifts below zero
        X = np.vstack([
            sample_actions(bounds=oracle.truth.bounds, n=N_PROBE, kind='uniform', seed=SEED),
            oracle.truth.optimizers.numpy()
        ])
        r = np.array([normalized_inference_regret(x, oracle, maximize=maximize) for x in X])

        assert r.min() >= -SCAN_SLACK
        assert r.max() <= 1 + SCAN_SLACK


class TestReturnValue:

    def test_the_truth_is_read_without_noise(self):
        spec  = dict(func='Levy', dim=3, box=BOX, seed=SEED)
        clean = SyntheticOracle.from_name(rel_noise_std=0.0, **spec)
        noisy = SyntheticOracle.from_name(rel_noise_std=0.5, **spec)
        action = np.full(3, 0.7)

        assert normalized_inference_regret(action, noisy) \
            == pytest.approx(normalized_inference_regret(action, clean))
        # a noiseless read leaves the oracle's noise stream where it found it
        assert normalized_inference_regret(action, noisy) \
            == normalized_inference_regret(action, noisy)

    def test_one_action_scores_a_scalar_however_it_is_shaped(self, levy):
        flat  = normalized_inference_regret(np.full(3, 0.7), levy)
        batch = normalized_inference_regret(np.full((1, 3), 0.7), levy)

        assert np.ndim(flat) == 0 and np.ndim(batch) == 0
        assert isinstance(flat, float)
        assert flat == pytest.approx(batch)
