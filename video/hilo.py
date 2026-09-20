"""Human-in-the-loop optimization: a device and an optimization exchanging feedback."""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    DoubleArrow,
    FadeIn,
    Group,
    FadeOut,
    GrowArrow,
    GrowFromCenter,
    Line,
    MathTex,
    Scene,
    Star,
    Tex,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the `video.style` import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.clip import VideoClip  # noqa: E402
from video.style import (  # noqa: E402
    ARROW_COLOR,
    AXIS_LABEL_SIZE,
    COMFORT_COLOR,
    COST_COLOR,
    DROP_COLOR,
    EQUATION_SIZE,
    INK,
    SAMPLE_COLOR,
    SCENE_TITLE_SIZE,
    STAR_COLOR,
    action_label,
    arrow_label,
    flow_arrow,
    labeled_box,
)

TITLE = "Human-in-the-loop Optimization (HILO)"
TITLE_EDGE_BUFF = 0.3

# Box geometry: a narrow device on the left, a wide optimization on the right,
# separated by a gap wide enough for the arrow labels. BOX_Y drops both boxes
# clear of the title.
DEVICE_W, DEVICE_H = 3.25, 5.6
OPT_W, OPT_H = 7.0, 5.6
DEVICE_X = -5.0
OPT_X = 2.85
BOX_Y = -0.35
# Two feedback arrows, one per objective, above the one carrying the controller back.
COST_FEEDBACK_Y = BOX_Y + 1.6
COMFORT_FEEDBACK_Y = BOX_Y + 0.55
PARAM_Y = BOX_Y - 1.1

# The subject clip shown in the device box. The crop is the ffmpeg w:h:x:y that
# removes the pillarbox bars around the portrait footage.
CLIP_PATH = Path(__file__).resolve().parent / "subjects" / "mb02" / "MB02_Overground_Portrait_blurred.mp4"
CLIP_CROP = "608:1080:656:0"
CLIP_H = 4.5
CLIP_Y = BOX_Y - 0.35

# True draws a cost bowl with its minimum starred; False flips it to a hump
# with its maximum starred.
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

# The second objective: the same parabola as a hump, peaking at a different
# action, so the cost optimum is not the comfort optimum.
COMFORT_OPTIMUM_A = 0.58

# Plot geometry, as offsets from the optimization box's center. AXIS_X_MAX runs
# the x axis past CURVE_X_RANGE so its tip clears the right-hand comfort axis,
# which stands at CURVE_X_RANGE[1].
PLOT_W, PLOT_H = 4.55, 3.1
PLOT_X = 0.1
PLOT_Y = -0.5
AXIS_X_MAX = 1.12
# Height of the gap arrow between the two optima, in curve units: above both
# curves, where the band between the two dashed lines is empty.
GAP_ARROW_Y = 1.07

# The scalarized objective, written in the band between the boxes and the frame
# edge, and the sweep of its weights over the simplex w_1 + w_2 = 1.
EQUATION_Y = -3.6
# The written width, which is what sets the type size: the equation is typeset at
# EQUATION_SIZE and then scaled to this.
EQUATION_W = 10.0
SWEEP_TIME = 2.0
SWEEP_LINE_WIDTH = 4.0

# The sampled actions the dot visits before landing on the optimum: the offset
# from OPTIMUM_A halves each step and falls on a random side of it.
N_SAMPLES = 5
SAMPLE_SPREAD = 0.3
SAMPLE_DECAY = 0.6
SAMPLE_SEED = 0
SAMPLE_MARGIN = 0.03
# Measurement noise on the sampled costs, in the units of the drawn curve.
SAMPLE_NOISE_STD = 0.07
TRAIL_OPACITY = 0.5


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


def sample_measurements():
    """N_SAMPLES actions closing in on OPTIMUM_A, and their noisy costs.

    Returns `(actions, values)`, both `(N_SAMPLES,)` and both kept inside the
    drawn axes.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    signs = rng.choice([-1.0, 1.0], size=N_SAMPLES)
    offsets = SAMPLE_SPREAD * SAMPLE_DECAY ** np.arange(N_SAMPLES) * signs
    actions = np.clip(
        OPTIMUM_A + offsets,
        CURVE_X_RANGE[0] + SAMPLE_MARGIN,
        CURVE_X_RANGE[1] - SAMPLE_MARGIN,
    )
    values = cost(actions) + rng.normal(0.0, SAMPLE_NOISE_STD, size=N_SAMPLES)
    return actions, np.clip(values, SAMPLE_MARGIN, CURVE_Y_MAX - SAMPLE_MARGIN)


class HiloScene(Scene):
    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        device = labeled_box("Device", [DEVICE_X, BOX_Y, 0], DEVICE_W, DEVICE_H)
        optimization = labeled_box("Optimization", [OPT_X, BOX_Y, 0], OPT_W, OPT_H)
        boxes = VGroup(device, optimization)

        clip = VideoClip(CLIP_PATH, CLIP_H, crop=CLIP_CROP)
        clip.move_to([DEVICE_X, CLIP_Y, 0])

        cost_feedback = flow_arrow(device, optimization, RIGHT, COST_FEEDBACK_Y, color=COST_COLOR)
        comfort_feedback = flow_arrow(
            device, optimization, RIGHT, COMFORT_FEEDBACK_Y, color=COMFORT_COLOR
        )
        parameterization = flow_arrow(optimization, device, LEFT, PARAM_Y)
        arrows = VGroup(cost_feedback, parameterization)
        labels = VGroup(
            arrow_label(["Metabolic Cost"], cost_feedback, color=COST_COLOR),
            arrow_label(["Controller", "Parameterization"], parameterization),
        )
        comfort_label = arrow_label(["Comfort"], comfort_feedback, color=COMFORT_COLOR)

        axes = self._axes(optimization)
        cost_labels, cost_curve, cost_star = self._objective(
            axes, "metabolic cost", cost, OPTIMUM_A, COST_COLOR, COST_WIDTH, convex=CONVEX
        )
        comfort_labels, comfort_curve, comfort_star = self._objective(
            axes, "comfort", comfort, COMFORT_OPTIMUM_A, COMFORT_COLOR, COMFORT_WIDTH,
            convex=not CONVEX, twin=True,
        )

        self.play(Write(title, run_time=1.0))
        self.play(FadeIn(Group(boxes, clip), run_time=1.2))
        clip.play(loop=True)
        self.wait(0.3)
        self.play(
            *(GrowArrow(arrow, run_time=0.6) for arrow in arrows),
            *(Write(label, run_time=0.8) for label in labels),
        )
        self.wait(0.3)
        self.play(Create(axes, run_time=0.8), Write(cost_labels, run_time=0.8))
        self.play(Create(cost_curve, run_time=1.0))
        samples = self._search(axes, cost_star)
        self.wait(0.4)
        self._second_objective(
            axes, comfort_labels, comfort_curve, comfort_star, comfort_feedback,
            comfort_label, samples,
        )
        self.wait(0.4)
        self._scalarize(axes)
        self.wait(max(0.5, clip.duration - self.renderer.time))

    def _search(self, axes, star):
        """Step a dot through the noisy measurements, marking each, and onto the star.

        Returns the dots left on the plot, which the second objective clears away.
        """
        actions, values = sample_measurements()
        action = ValueTracker(actions[0])

        scan = always_redraw(
            lambda: DashedLine(
                axes.c2p(action.get_value(), 0.0),
                axes.c2p(action.get_value(), CURVE_Y_MAX),
                color=DROP_COLOR,
                stroke_width=2.5,
                dash_length=0.08,
            )
        )
        dot = Dot(axes.c2p(actions[0], values[0]), radius=0.09, color=SAMPLE_COLOR)
        samples = VGroup(dot)

        self.play(Create(scan, run_time=0.4), FadeIn(dot, scale=0.5, run_time=0.4))
        for a, y in zip(actions[1:], values[1:]):
            samples.add(dot.copy().set_opacity(TRAIL_OPACITY))
            self.add(samples)
            self.play(
                action.animate.set_value(a),
                dot.animate.move_to(axes.c2p(a, y)),
                run_time=0.7,
            )
        samples.add(dot.copy().set_opacity(TRAIL_OPACITY))
        self.add(samples)
        self.play(
            action.animate.set_value(OPTIMUM_A),
            dot.animate.move_to(axes.c2p(OPTIMUM_A, cost(OPTIMUM_A))),
            run_time=0.9,
        )
        self.play(FadeIn(star, scale=0.4, run_time=0.6))
        return samples

    def _second_objective(
        self, axes, comfort_labels, comfort_curve, comfort_star, feedback, feedback_label,
        samples,
    ):
        """Draw comfort against the right-hand axis and mark what the cost optimum costs it.

        A second feedback arrow arrives from the device first, since comfort is
        another measurement the device sends. The scan line already spans the plot
        at OPTIMUM_A, so once the comfort curve is drawn the line crosses it, away
        from its own optimum, with no marker of its own. The measured points go
        with that curve: what is left is the two optima and the gap between them.
        """
        self.play(GrowArrow(feedback, run_time=0.6), Write(feedback_label, run_time=0.8))
        self.play(Create(comfort_labels[0], run_time=0.6), Write(comfort_labels[1], run_time=0.6))
        self.play(Create(comfort_curve, run_time=1.0), FadeOut(samples, run_time=1.0))

        optimum_drop = DashedLine(
            axes.c2p(COMFORT_OPTIMUM_A, 0.0),
            axes.c2p(COMFORT_OPTIMUM_A, CURVE_Y_MAX),
            color=DROP_COLOR,
            stroke_width=2.5,
            dash_length=0.08,
        )
        self.play(FadeIn(comfort_star, scale=0.4, run_time=0.6), Create(optimum_drop, run_time=0.6))

        gap = DoubleArrow(
            axes.c2p(OPTIMUM_A, GAP_ARROW_Y),
            axes.c2p(COMFORT_OPTIMUM_A, GAP_ARROW_Y),
            buff=0.0,
            color=ARROW_COLOR,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.12,
        )
        self.play(GrowFromCenter(gap, run_time=0.6))

    def _scalarize(self, axes):
        """Write the scalarized objective, then sweep its weights across the simplex.

        `scalarized_argmin` is the argmin of the written objective itself, so the
        solid line is where that objective is minimized rather than an interpolation
        between the two stars; it leaves the cost optimum only as w_2 grows, and
        the two readouts always sum to one.
        """
        weight = ValueTracker(1.0)
        equation = self._scalarized_equation(weight)
        sweep = always_redraw(
            lambda: Line(
                axes.c2p(scalarized_argmin(weight.get_value()), 0.0),
                axes.c2p(scalarized_argmin(weight.get_value()), CURVE_Y_MAX),
                color=INK,
                stroke_width=SWEEP_LINE_WIDTH,
            )
        )

        self.play(Write(equation, run_time=1.4))
        self.play(Create(sweep, run_time=0.5))
        self.play(weight.animate.set_value(0.0), run_time=SWEEP_TIME)
        self.wait(0.4)
        self.play(weight.animate.set_value(1.0), run_time=SWEEP_TIME)

    def _scalarized_equation(self, weight):
        """The scalarized objective, with a `w` readout driven by `weight`.

        The weights are `DecimalNumber`s rather than part of the LaTeX, so the
        sweep re-renders two cached digit glyphs per frame instead of compiling a
        new equation.
        """
        lhs = MathTex(
            r"\bm{a}_{\mathrm{scalarized}} = \argmin_{\bm{a} \in \mathcal{A}}\;",
            r"w_1 f_{\mathrm{Metabolic}}",
            "-",
            r"w_2 f_{\mathrm{Comfort}}",
            ",",
            font_size=EQUATION_SIZE,
            color=INK,
        )
        lhs[1].set_color(COST_COLOR)
        lhs[3].set_color(COMFORT_COLOR)

        # One MathTex, so the brackets and comma sit on the same baseline as the
        # numbers; its two slots are typeset and then hidden under the readouts.
        slots = MathTex(
            r"\bm{w} = [", "1.00", r",\;", "0.00", "]",
            font_size=EQUATION_SIZE,
            color=INK,
        )
        w1 = DecimalNumber(1.0, num_decimal_places=2, font_size=EQUATION_SIZE, color=COST_COLOR)
        w2 = DecimalNumber(0.0, num_decimal_places=2, font_size=EQUATION_SIZE, color=COMFORT_COLOR)
        w1.move_to(slots[1])
        w2.move_to(slots[3])
        slots[1].set_opacity(0.0)
        slots[3].set_opacity(0.0)
        w1.add_updater(lambda d: d.set_value(weight.get_value()))
        w2.add_updater(lambda d: d.set_value(1.0 - weight.get_value()))
        readout = VGroup(slots, w1, w2)

        equation = VGroup(lhs, readout).arrange(RIGHT, buff=0.45)
        equation.scale_to_fit_width(EQUATION_W).move_to([0, EQUATION_Y, 0])
        return equation

    def _axes(self, box):
        """The shared axes: one x axis in action, one left y axis, sized to sit in the box."""
        axes = Axes(
            x_range=[CURVE_X_RANGE[0], AXIS_X_MAX, 1.0],
            y_range=[0.0, CURVE_Y_MAX, 1.0],
            x_length=PLOT_W,
            y_length=PLOT_H,
            axis_config={
                "color": INK,
                "stroke_width": 2.5,
                "include_ticks": False,
                "tip_length": 0.18,
                "tip_width": 0.18,
            },
        )
        axes.move_to(box[0].get_center() + np.array([PLOT_X, PLOT_Y, 0]))
        axes.y_axis.set_color(COST_COLOR)
        return axes

    def _objective(self, axes, label, func, optimum, color, width, convex=True, twin=False):
        """One objective on `axes`: its labels, curve and starred optimum, all in `color`.

        `twin=False` labels the left y axis, which `_axes` already drew; `twin=True`
        adds a second y axis at the right end of the x axis and labels that, so the
        two objectives share one x axis and read against their own y axis.
        """
        curve = axes.plot(
            func,
            x_range=curve_domain(optimum, width, convex=convex),
            color=color,
            stroke_width=5.0,
        )
        star = Star(
            n=5,
            outer_radius=0.16,
            inner_radius=0.07,
            color=STAR_COLOR,
            fill_color=STAR_COLOR,
            fill_opacity=1.0,
            stroke_width=1.0,
        ).move_to(axes.c2p(optimum, func(optimum)))

        text = Tex(label, font_size=AXIS_LABEL_SIZE, color=color)
        if twin:
            axis = axes.y_axis.copy().set_color(color)
            axis.shift([axes.c2p(CURVE_X_RANGE[1], 0.0)[0] - axes.c2p(0.0, 0.0)[0], 0, 0])
            text.rotate(-np.pi / 2).next_to(axis, RIGHT, buff=0.18)
            labels = VGroup(axis, text)
        else:
            text.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
            labels = VGroup(action_label().next_to(axes.x_axis, DOWN, buff=0.18), text)

        return labels, curve, star
