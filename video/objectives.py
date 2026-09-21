"""The two objectives every scene in this package draws, as one pypolar groundtruth.

Metabolic cost is a bowl and comfort a hump, both `IdealPoint`s squashed by a tanh
into a band, so the action that minimizes cost is not the one that maximizes comfort.
They are built here as a `MOSyntheticOracle` rather than written out as curves,
which is what lets a scene both *draw* them noiselessly and *measure* them with
noise from one definition -- and lets `mogp.py` fit the repo's own
`DecoupledMOGP` to those measurements. `hilo/shared/simulation.py` builds the live
experiment's groundtruth the same way, in three dimensions instead of one.
"""

import numpy as np
import pypolar as plr
from scipy.optimize import minimize_scalar

COST_NAME = 'Metabolic Cost'
COMFORT_NAME = 'Comfort'

CURVE_X_RANGE = (0.0, 1.0)
OPTIMUM_A = 0.25
COMFORT_OPTIMUM_A = 0.75

# The band each bowl is squashed into, and its curvature before the squash. The
# comfort weight is negative, which is what flips its bowl into a hump: the tanh
# then puts CURVE_PEAK at its optimum instead of its floor. The two magnitudes
# differ so that the objectives do not saturate in lockstep, which is what bows
# the front rather than leaving it a straight diagonal.
COST_BAND = (0.12, 1.0)
COMFORT_BAND = (0.0, 1.0)
COST_WEIGHT = 16.0
COMFORT_WEIGHT = -6.0

CURVE_FLOOR = COST_BAND[0]
CURVE_PEAK = max(COST_BAND[1], COMFORT_BAND[1])
# Headroom above the band, for the axis tips. Nothing is ever clipped to it:
# a bounded bowl cannot leave its own band.
CURVE_Y_MAX = CURVE_PEAK * 1.15

# Observation noise, as a fraction of each objective's own spread, and the seed
# its draws come from.
NOISE_STD = 0.25
SEED = 0

TRUTH_PARAMS = plr.SyntheticOracleParams(
    func          = 'IdealPoint',
    objectives    = (COST_NAME, COMFORT_NAME),
    dim           = 1,
    box           = CURVE_X_RANGE,
    optima        = ((OPTIMUM_A,), (COMFORT_OPTIMUM_A,)),
    weights       = (COST_WEIGHT, COMFORT_WEIGHT),
    low           = (COST_BAND[0], COMFORT_BAND[0]),
    high          = (COST_BAND[1], COMFORT_BAND[1]),
    rel_noise_std = NOISE_STD,
    seed          = SEED,
    measure       = 'std',
)

TRUTH = TRUTH_PARAMS.build()
COST_TRUTH = TRUTH.get_oracle(0).truth
COMFORT_TRUTH = TRUTH.get_oracle(1).truth


def _actions(a):
    """`a` as the `(n, 1)` array a truth takes, clipped into the action box.

    Both halves are load-bearing. A `SyntheticTestFunction` reads its action
    dimension off the last axis, so a bare float raises rather than broadcasting,
    and manim's `axes.plot` calls an objective one float at a time. It also
    rejects actions outside its bounds, and stepping a float `x_range` overshoots
    the end of the box by an epsilon.
    """
    return np.clip(np.reshape(np.asarray(a, dtype=float), (-1, 1)), *CURVE_X_RANGE)


def _values(truth, a):
    """That truth's noiseless values at `a`, shaped like `a` was."""
    values = plr.truth_at(truth, _actions(a))
    return float(values[0]) if np.ndim(a) == 0 else values


def cost(a):
    """The metabolic cost curve: a bowl bottoming out at OPTIMUM_A."""
    return _values(COST_TRUTH, a)


def comfort(a):
    """The comfort curve: a hump peaking at COMFORT_OPTIMUM_A."""
    return _values(COMFORT_TRUTH, a)


def measure(actions):
    """Both objectives measured with noise at the same actions, `(n, 2)`.

    One call for both columns, so a pair of measurements really is one visit to
    one action, and the draws advance a single stream rather than two.
    """
    return TRUTH(_actions(actions))


def sobol_actions(n, seed=SEED):
    """`n` actions spanning the box, `(n, 1)`, space-filling rather than clumped."""
    return plr.sample_actions(bounds=TRUTH.bounds, n=n, kind='sobol', seed=seed)


def attainable_actions():
    """The actions a curve is drawn over: the whole box.

    A bounded bowl stays inside its band everywhere, so unlike an unbounded one
    there is no sub-interval to restrict a curve to.
    """
    return list(CURVE_X_RANGE)


def front_actions():
    """The actions whose objective values are Pareto optimal.

    Below OPTIMUM_A both objectives are worse than at OPTIMUM_A, and above
    COMFORT_OPTIMUM_A both are worse than there, so every non-dominated point
    comes from between the two single-objective optima. The tanh is monotone in
    the squared distance from an optimum, so squashing the bowls does not move
    that interval.
    """
    return [OPTIMUM_A, COMFORT_OPTIMUM_A]


def scalarized_argmin(w1):
    """The action minimizing `w1 * cost - (1 - w1) * comfort`, with `w` on the simplex.

    Solved numerically rather than in closed form: the two curves are tanh-squashed
    bowls rather than parabolas, so their weighted difference is no longer a
    quadratic with an analytic minimum. Brent on the bounded interval is the right
    tool because the sum has a single interior minimum, which was checked against a
    4001-point grid argmin over the whole simplex -- the two agree to the grid's own
    resolution, and the path from OPTIMUM_A to COMFORT_OPTIMUM_A is monotone.
    """
    scalarized = lambda a: w1 * cost(a) - (1.0 - w1) * comfort(a)
    return minimize_scalar(scalarized, bounds=CURVE_X_RANGE, method='bounded').x
