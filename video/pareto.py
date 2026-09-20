"""Pareto optimality: the two objectives in action space, and their front in objective space.

The left column plots each objective against the action. The right panel plots
them against each other, which is where "Pareto optimal" is visible: on the front,
the only way to gain on one objective is to give up on the other, drawn here as a
green leg and a red leg between two controllers.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    UP,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
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
    CURVE_PEAK,
    CURVE_Y_MAX,
    OPTIMUM_A,
    attainable_actions,
    comfort,
    comfort_domain,
    cost,
    cost_domain,
)
from video.style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    COMFORT_COLOR,
    COST_COLOR,
    DROP_COLOR,
    GAIN_COLOR,
    INK,
    LOSS_COLOR,
    SCENE_TITLE_SIZE,
    action_label,
)

TITLE = "Pareto Optimality"
TITLE_EDGE_BUFF = 0.3

# Left column: the two objectives against the action, stacked and sharing an x
# axis, so one dashed line at an action crosses both.
ACTION_W, ACTION_H = 4.0, 2.15
ACTION_X = -4.55
METABOLIC_Y = 1.15
COMFORT_Y = -1.75
AXIS_X_MAX = 1.05

# Right panel: the objective space, cost across and comfort up. Both attainable
# ranges are known from the curves, so the axes only add headroom for the tips.
FRONT_W, FRONT_H = 5.3, 4.9
FRONT_X = 3.0
FRONT_Y = -0.3
FRONT_X_MAX = CURVE_Y_MAX * 1.12
FRONT_Y_MAX = CURVE_PEAK * 1.15

# The two controllers compared on the front, both between the single-objective
# optima so that neither dominates the other.
A1, A2 = 0.35, 0.53

CURVE_STROKE = 5.0
LANDSCAPE_STROKE = 3.5
PLOT_SAMPLES = 0.004

# Where the label naming the front sits, in objective units. Its arrow lands on
# the middle of the front, which is computed rather than placed.
POINTER_AT = (0.47, 0.72)

# The dotted legs between the two controllers.
LEG_DASH = 0.07
LEG_STROKE = 4.0
LEG_TIP = 0.22


def front_actions():
    """The actions whose objective values are Pareto optimal.

    Below OPTIMUM_A both objectives are worse than at OPTIMUM_A, and above
    COMFORT_OPTIMUM_A both are worse than there, so every non-dominated point comes
    from between the two single-objective optima.
    """
    return [OPTIMUM_A, COMFORT_OPTIMUM_A]


class ParetoScene(Scene):
    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        metabolic_axes = self._action_axes(METABOLIC_Y)
        comfort_axes = self._action_axes(COMFORT_Y)
        metabolic = self._action_plot(
            metabolic_axes, "metabolic cost", cost, cost_domain(), COST_COLOR
        )
        comfort_plot = self._action_plot(
            comfort_axes, "comfort", comfort, comfort_domain(), COMFORT_COLOR, x_label=True
        )
        front_axes, front_labels = self._front_axes()
        landscape, front = self._front_curves(front_axes)
        pointer, pointer_label = self._front_pointer(front_axes)

        self.play(Write(title, run_time=1.0))
        self.play(
            Create(metabolic_axes, run_time=0.7),
            Create(comfort_axes, run_time=0.7),
            Write(metabolic[0], run_time=0.7),
            Write(comfort_plot[0], run_time=0.7),
        )
        self.play(Create(metabolic[1], run_time=0.9), Create(comfort_plot[1], run_time=0.9))
        self.wait(0.4)

        self.play(Create(front_axes, run_time=0.8), Write(front_labels, run_time=0.8))
        self.play(Create(landscape, run_time=1.6))
        self.play(Create(front, run_time=1.2))
        self.play(GrowArrow(pointer, run_time=0.6), Write(pointer_label, run_time=0.7))
        self.wait(0.4)

        self._mark(A1, r"\bm{a}_1", metabolic_axes, comfort_axes, front_axes)
        self.wait(0.3)
        self._tradeoff(front_axes)
        self._mark(A2, r"\bm{a}_2", metabolic_axes, comfort_axes, front_axes)
        self.wait(1.0)

    def _action_axes(self, y):
        """Axes for one objective against the action, in the left column."""
        axes = Axes(
            x_range=[0.0, AXIS_X_MAX, 1.0],
            y_range=[0.0, CURVE_Y_MAX, 1.0],
            x_length=ACTION_W,
            y_length=ACTION_H,
            axis_config={
                "color": INK,
                "stroke_width": 2.5,
                "include_ticks": False,
                "tip_length": 0.18,
                "tip_width": 0.18,
            },
        )
        axes.move_to([ACTION_X, y, 0])
        return axes

    def _action_plot(self, axes, label, func, domain, color, x_label=False):
        """That objective's labels and curve, the labels in its own color.

        Only the lower plot is given the action label, since the two share an x
        axis; it goes at the right end, leaving the space under the axis for the
        controllers' own labels.
        """
        axes.y_axis.set_color(color)
        text = Tex(label, font_size=AXIS_LABEL_SIZE, color=color)
        text.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
        labels = VGroup(text)
        if x_label:
            labels.add(action_label().next_to(axes.x_axis.get_end(), DOWN, buff=0.2))
        curve = axes.plot(func, x_range=domain, color=color, stroke_width=CURVE_STROKE)
        return VGroup(labels, curve)

    def _front_axes(self):
        """The objective space: metabolic cost across, comfort up."""
        axes = Axes(
            x_range=[0.0, FRONT_X_MAX, 1.0],
            y_range=[0.0, FRONT_Y_MAX, 1.0],
            x_length=FRONT_W,
            y_length=FRONT_H,
            axis_config={
                "color": INK,
                "stroke_width": 2.5,
                "include_ticks": False,
                "tip_length": 0.18,
                "tip_width": 0.18,
            },
        )
        axes.move_to([FRONT_X, FRONT_Y, 0])
        axes.x_axis.set_color(COST_COLOR)
        axes.y_axis.set_color(COMFORT_COLOR)

        x_label = Tex("metabolic cost", font_size=AXIS_LABEL_SIZE, color=COST_COLOR)
        x_label.next_to(axes.x_axis, DOWN, buff=0.18)
        y_label = Tex("comfort", font_size=AXIS_LABEL_SIZE, color=COMFORT_COLOR)
        y_label.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
        return axes, VGroup(x_label, y_label)

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
        """One controller, drawn in all three plots at once.

        The two left plots share an x axis, so a single dashed line at `action`
        crosses both; the point it names in objective space is the pair of values
        the two dots read off.
        """

        drop = DashedLine(
            metabolic_axes.c2p(action, CURVE_Y_MAX),
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

    def _tradeoff(self, axes):
        """The two legs from a_1 to a_2: first what it gains, then what that costs.

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

    def _leg(self, start, end, color):
        """One dotted leg with a solid head, since a dashed tip reads as a smudge."""
        leg = DashedLine(
            start, end, dash_length=LEG_DASH, color=color, stroke_width=LEG_STROKE
        )
        leg.add_tip(tip_length=LEG_TIP, tip_width=LEG_TIP)
        return leg
