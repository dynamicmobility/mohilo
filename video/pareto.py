"""Pareto optimality: the two objectives in action space, and their front in objective space.

The left column plots each objective against the action. The right panel plots
them against each other, which is where "Pareto optimal" is visible: on the front,
the only way to gain on one objective is to give up on the other, drawn here as a
green leg and a red leg between two controllers.
"""

import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    UP,
    Arrow,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    MathTex,
    Scene,
    Tex,
    VGroup,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the `video.style` import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.objectives import (  # noqa: E402
    COMFORT_OPTIMUM_A,
    attainable_actions,
    comfort,
    cost,
    front_actions,
)
from video.style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    AXIS_Y_MAX,
    COMFORT_COLOR,
    COMFORT_Y,
    COST_COLOR,
    CURVE_STROKE,
    DROP_COLOR,
    GAIN_COLOR,
    INK,
    LANDSCAPE_STROKE,
    LOSS_COLOR,
    METABOLIC_Y,
    PLOT_SAMPLES,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
    action_axes,
    action_plot,
    front_axes as front_axes_builder,
)

TITLE = "Pareto Optimality"

# The two controllers compared on the front, both between the single-objective
# optima so that neither dominates the other.
A1, A2 = 0.34, 0.66

# Where the label naming the front sits, in objective units. Its arrow lands on
# the middle of the front, which is computed rather than placed.
POINTER_AT = (0.47, 0.72)

# The dotted legs between the two controllers.
LEG_DASH = 0.07
LEG_STROKE = 4.0
LEG_TIP = 0.22


class ParetoScene(Scene):
    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        metabolic_axes = action_axes(METABOLIC_Y)
        comfort_axes = action_axes(COMFORT_Y)
        metabolic = action_plot(
            metabolic_axes, "metabolic cost", cost, attainable_actions(), COST_COLOR
        )
        comfort_plot = action_plot(
            comfort_axes, "comfort", comfort, attainable_actions(), COMFORT_COLOR,
            x_label=True
        )
        front_axes, front_labels = front_axes_builder()
        landscape, front = self._front_curves(front_axes)
        pointer, pointer_label = self._front_pointer(front_axes)

        self.play(Write(title, run_time=1.0))
        # All three axes together, then all three curves together. Each curve is
        # drawn from the smallest action to the largest, so the two action plots
        # sweep left to right while the landscape traces that same sweep through
        # objective space -- one action being swept, seen three ways.
        self.play(
            Create(metabolic_axes, run_time=0.8),
            Create(comfort_axes, run_time=0.8),
            Create(front_axes, run_time=0.8),
            Write(metabolic[0], run_time=0.8),
            Write(comfort_plot[0], run_time=0.8),
            Write(front_labels, run_time=0.8),
        )
        self.play(
            Create(metabolic[1], run_time=1.6),
            Create(comfort_plot[1], run_time=1.6),
            Create(landscape, run_time=1.6),
        )
        self.wait(0.4)

        self.play(Create(front, run_time=1.2))
        self.play(GrowArrow(pointer, run_time=0.6), Write(pointer_label, run_time=0.7))
        self.wait(0.4)

        first = self._mark(A1, r"\bm{a}_1", metabolic_axes, comfort_axes, front_axes)
        self.wait(0.3)
        legs = self._tradeoff(front_axes)
        second = self._mark(A2, r"\bm{a}_2", metabolic_axes, comfort_axes, front_axes)
        self.wait(1.0)

        # What `MogpScene` opens on: the axes, their labels, the true curves and the
        # attainable landscape, and nothing else. Everything this scene drew over
        # them goes, so the two scenes cut together with no redraw.
        self.play(FadeOut(VGroup(title, front, pointer, pointer_label,
                                 first, legs, second), run_time=0.8))
        self.wait(0.4)

    def _front_curves(self, axes):
        """The whole attainable landscape, and the front that is drawn over it.

        Both are the same parametric curve `(cost(a), comfort(a))` and only the
        range of actions differs, so the bright arc lands exactly on the stretch of
        the dim one it replaces.
        """
        low, high = attainable_actions()
        start, end = front_actions()
        landscape = axes.plot_parametric_curve(
            lambda a: (cost(a), comfort(a)),
            t_range=[low, high, PLOT_SAMPLES],
            color=DROP_COLOR,
            stroke_width=LANDSCAPE_STROKE,
        )
        front = axes.plot_parametric_curve(
            lambda a: (cost(a), comfort(a)),
            t_range=[start, end, PLOT_SAMPLES],
            color=INK,
            stroke_width=CURVE_STROKE,
        )
        return landscape, front

    def _front_pointer(self, axes):
        """The name of the bright arc, and the arrow from it to the middle of that arc.

        The tip is the front's own midpoint in action, so it follows the curve if
        either optimum moves rather than being a placed coordinate.
        """
        label = MathTex(r"\mathcal{F}", font_size=AXIS_LABEL_SIZE, color=INK)
        label.move_to(axes.c2p(*POINTER_AT))
        middle = sum(front_actions()) / 2.0
        arrow = Arrow(
            label.get_corner(UP + LEFT),
            axes.c2p(cost(middle), comfort(middle)),
            buff=0.12,
            color=INK,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.15,
        )
        return arrow, label

    def _mark(self, action, label, metabolic_axes, comfort_axes, front_axes):
        """One controller, drawn in all three plots at once, and returned.

        The two left plots share an x axis, so a single dashed line at `action`
        crosses both; the point it names in objective space is the pair of values
        the two dots read off. Returned because `MogpScene` opens on the bare
        axes, so everything drawn over them here is cleared before the cut.
        """

        drop = DashedLine(
            metabolic_axes.c2p(action, AXIS_Y_MAX),
            comfort_axes.c2p(action, 0.0),
            color=DROP_COLOR,
            stroke_width=2.5,
            dash_length=0.08,
        )
        dots = VGroup(
            Dot(metabolic_axes.c2p(action, cost(action)), radius=0.08, color=INK),
            Dot(comfort_axes.c2p(action, comfort(action)), radius=0.08, color=INK),
        )
        action_text = MathTex(label, font_size=AXIS_LABEL_SIZE, color=INK)
        action_text.next_to(comfort_axes.c2p(action, 0.0), DOWN, buff=0.15)

        point = Dot(front_axes.c2p(cost(action), comfort(action)), radius=0.09, color=INK)
        point_text = MathTex(label, font_size=AXIS_LABEL_SIZE, color=INK)
        point_text.next_to(point, LEFT if action < COMFORT_OPTIMUM_A / 2 + 0.1 else UP, buff=0.12)

        self.play(
            Create(drop, run_time=0.5),
            FadeIn(dots, scale=0.5, run_time=0.5),
            Write(action_text, run_time=0.5),
        )
        self.play(FadeIn(point, scale=0.5, run_time=0.4), Write(point_text, run_time=0.5))
        return VGroup(drop, dots, action_text, point, point_text)

    def _tradeoff(self, axes):
        """The two legs from a_1 to a_2, returned: first what it gains, then what
        that costs.

        They are the sides of the right triangle spanning the two points, so each
        leg is the change in one objective on its own. Going up first turns the
        corner at `(cost(a_1), comfort(a_2))` — more comfort at a_1's cost, which
        no controller attains — so the red leg is the walk back to the front, and
        it ends exactly where a_2 is drawn next. That is the claim: the gain is
        only available if the loss is paid.
        """
        first = axes.c2p(cost(A1), comfort(A1))
        second = axes.c2p(cost(A2), comfort(A2))
        corner = axes.c2p(cost(A1), comfort(A2))

        gain = self._leg(first, corner, GAIN_COLOR)
        loss = self._leg(corner, second, LOSS_COLOR)
        self.play(Create(gain, run_time=0.7))
        self.wait(0.2)
        self.play(Create(loss, run_time=0.7))
        return VGroup(gain, loss)

    def _leg(self, start, end, color):
        """One dotted leg with a solid head, since a dashed tip reads as a smudge."""
        leg = DashedLine(
            start, end, dash_length=LEG_DASH, color=color, stroke_width=LEG_STROKE
        )
        leg.add_tip(tip_length=LEG_TIP, tip_width=LEG_TIP)
        return leg
