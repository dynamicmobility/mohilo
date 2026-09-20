"""The two objectives every scene in this package draws, as plain functions of the action.

Metabolic cost is a bowl and comfort a hump, both parabolas about their own
optimum, so the action that minimizes cost is not the one that maximizes comfort.
Nothing here touches manim: a scene decides how to draw these, and both the HILO
loop and the Pareto front read the same curves.
"""

import numpy as np

# True draws a cost bowl with its minimum at OPTIMUM_A; False flips both curves.
CONVEX = True

CURVE_X_RANGE = (0.0, 1.0)
CURVE_Y_MAX = 1.15
OPTIMUM_A = 0.3
CURVE_FLOOR = 0.12
CURVE_PEAK = 1.0
# Narrows the parabolas: each is as wide as this fraction of the span that would
# put CURVE_PEAK at the far end of the x range. Comfort is the wider of the two,
# which is what keeps its value at the cost optimum clear of the cost curve.
COST_WIDTH = 0.55
COMFORT_WIDTH = 0.72

# The second objective peaks at a different action, so the two disagree.
COMFORT_OPTIMUM_A = 0.58


def reach(optimum, width):
    """The half-width over which the parabola about `optimum` rises by its full range."""
    return width * max(optimum - CURVE_X_RANGE[0], CURVE_X_RANGE[1] - optimum)


def parabola(a, optimum, width, convex=True):
    """A parabola about `optimum`, rising from CURVE_FLOOR at a rate set by `reach`.

    At `width = 1` it reaches CURVE_PEAK at whichever end of the x range is
    further from the optimum; narrower than that it leaves the axes before then,
    and `curve_domain` is what keeps it on screen. `convex=False` flips the bowl
    into a hump, putting CURVE_PEAK at the optimum.
    """
    scale = (CURVE_PEAK - CURVE_FLOOR) / reach(optimum, width) ** 2
    value = scale * (a - optimum) ** 2 + CURVE_FLOOR
    return value if convex else CURVE_PEAK + CURVE_FLOOR - value


def curve_domain(optimum, width, convex=True):
    """The x interval over which that parabola stays inside the axes.

    A bowl is drawn up to CURVE_Y_MAX and a hump down to the x axis, so each
    curve ends at an edge of the plot rather than being clipped flat against it.
    """
    headroom = CURVE_Y_MAX - CURVE_FLOOR if convex else CURVE_PEAK
    half = reach(optimum, width) * np.sqrt(headroom / (CURVE_PEAK - CURVE_FLOOR))
    return [
        max(CURVE_X_RANGE[0], optimum - half),
        min(CURVE_X_RANGE[1], optimum + half),
    ]


def curvature(optimum, width):
    """The multiplier on `(a - optimum) ** 2` in that parabola."""
    return (CURVE_PEAK - CURVE_FLOOR) / reach(optimum, width) ** 2


def scalarized_argmin(w1):
    """The action minimizing `w1 * cost - (1 - w1) * comfort`, with `w` on the simplex.

    Cost is a bowl `kc (a - ac)^2` and comfort a hump `-kf (a - af)^2`, both up to
    a constant, so subtracting the hump leaves a sum of two upward parabolas. Its
    minimum is where the derivative vanishes, at the curvature-weighted average of
    the two optima, which runs from `ac` at `w1 = 1` to `af` at `w1 = 0`.
    """
    cost_weight = w1 * curvature(OPTIMUM_A, COST_WIDTH)
    comfort_weight = (1.0 - w1) * curvature(COMFORT_OPTIMUM_A, COMFORT_WIDTH)
    numerator = cost_weight * OPTIMUM_A + comfort_weight * COMFORT_OPTIMUM_A
    return numerator / (cost_weight + comfort_weight)


def cost(a):
    """The metabolic cost curve: a bowl bottoming out at OPTIMUM_A."""
    return parabola(a, OPTIMUM_A, COST_WIDTH, convex=CONVEX)


def comfort(a):
    """The comfort curve: a hump peaking at COMFORT_OPTIMUM_A."""
    return parabola(a, COMFORT_OPTIMUM_A, COMFORT_WIDTH, convex=not CONVEX)


def cost_domain():
    """The actions over which the cost bowl is inside its axes."""
    return curve_domain(OPTIMUM_A, COST_WIDTH, convex=CONVEX)


def comfort_domain():
    """The actions over which the comfort hump is inside its axes."""
    return curve_domain(COMFORT_OPTIMUM_A, COMFORT_WIDTH, convex=not CONVEX)


def attainable_actions():
    """The actions where *both* curves are on screen, which is the attainable set.

    A point of the objective space is only drawable if both of its coordinates are,
    so the Pareto front is the image of this interval rather than of the whole x
    range. By construction of `curve_domain` its ends are where cost hits
    CURVE_Y_MAX and comfort hits zero.
    """
    low = max(cost_domain()[0], comfort_domain()[0])
    high = min(cost_domain()[1], comfort_domain()[1])
    return [low, high]
