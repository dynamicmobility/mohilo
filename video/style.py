"""Shared palette, geometry and manim config for every scene in this package.

Importing this module sets the render background, the LaTeX template and the
media directory, so every scene looks the same without repeating the
configuration. It also owns the two-panel layout `ParetoScene` and `MogpScene`
share: that scene cuts straight into this one, on the same axes, so the geometry
and the builders are stated once here rather than copied into both.
"""

from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    Arrow,
    Axes,
    ManimColor,
    MathTex,
    RoundedRectangle,
    Tex,
    TexTemplate,
    VGroup,
    config,
)

from video.objectives import CURVE_Y_MAX

# Palette: dark ink on white, one color per objective, one accent for the optimum,
# and a green/red pair kept free to mean gain and loss.
BG = ManimColor("#FFFFFF")
INK = ManimColor("#0D0D0D")
BOX_FILL = ManimColor("#EDE8E2")
BOX_STROKE = ManimColor("#8AA0B4")
ARROW_COLOR = ManimColor("#C7D3DE")
COST_COLOR = ManimColor("#0F7DA8")
COMFORT_COLOR = ManimColor("#8A4FD6")
STAR_COLOR = ManimColor("#B87A00")
GAIN_COLOR = ManimColor("#4FD17A")
LOSS_COLOR = ManimColor("#F25F5C")
SAMPLE_COLOR = ManimColor("#0D0D0D")
DROP_COLOR = ManimColor("#6E7B87")

SCENE_TITLE_SIZE = 54
TITLE_SIZE = 44
ARROW_LABEL_SIZE = 28
AXIS_LABEL_SIZE = 30
EQUATION_SIZE = 40

# Box geometry, in manim scene units, inside the 14.2 x 8.0 frame.
BOX_RADIUS = 0.18
BOX_STROKE_WIDTH = 3.0
TITLE_INSET = 0.45

ARROW_BUFF = 0.0
ARROW_STROKE_WIDTH = 5.0
ARROW_TIP_LENGTH = 0.28

# Computer Modern throughout, with bm for bold symbols and argmin as an operator,
# which amsmath provides but does not define.
TEX_TEMPLATE = TexTemplate()
TEX_TEMPLATE.add_to_preamble(r"\usepackage{bm}")
TEX_TEMPLATE.add_to_preamble(r"\DeclareMathOperator*{\argmin}{arg\,min}")

config.background_color = BG
config.tex_template = TEX_TEMPLATE
config.media_dir = str(Path(__file__).resolve().parent / "media")


def labeled_box(label, center, width, height):
    """A rounded box with its title near the top edge, as a VGroup(rect, tex)."""
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=BOX_RADIUS,
        stroke_color=BOX_STROKE,
        stroke_width=BOX_STROKE_WIDTH,
        fill_color=BOX_FILL,
        fill_opacity=1.0,
    ).move_to(center)
    text = Tex(label, font_size=TITLE_SIZE, color=INK)
    text.move_to(rect.get_top() + [0, -TITLE_INSET, 0])
    return VGroup(rect, text)


def flow_arrow(start_box, end_box, direction, y, color=ARROW_COLOR):
    """A horizontal arrow at height y, spanning the gap between two boxes' facing edges."""
    start = start_box[0].get_edge_center(direction)
    end = end_box[0].get_edge_center(-direction)
    return Arrow(
        start=[start[0], y, 0],
        end=[end[0], y, 0],
        buff=ARROW_BUFF,
        color=color,
        stroke_width=ARROW_STROKE_WIDTH,
        max_tip_length_to_length_ratio=1.0,
        tip_length=ARROW_TIP_LENGTH,
    )


def arrow_label(lines, arrow, buff=0.12, color=INK):
    """One or more lines of text stacked above an arrow, centered on its span."""
    text = VGroup(*(Tex(line, font_size=ARROW_LABEL_SIZE, color=color) for line in lines))
    text.arrange(DOWN, buff=0.10)
    text.next_to(arrow.get_center(), -DOWN, buff=buff)
    return text


def action_label(font_size=AXIS_LABEL_SIZE):
    """The action symbol, \\bm{a}."""
    return MathTex(r"\bm{a}", font_size=font_size, color=INK)


# The two-panel layout, in manim scene units. Left column: the two objectives
# against the action, stacked and sharing an x axis, so one line at an action
# crosses both. Right panel: the objective space, cost across and comfort up.
ACTION_W, ACTION_H = 4.0, 2.15
ACTION_X = -4.55
METABOLIC_Y = 0.95
COMFORT_Y = -1.95
AXIS_X_MAX = 1.05
# Taller than the band the curves live in, since a measurement at MogpScene's
# noise lands well outside that band.
AXIS_Y_MAX = CURVE_Y_MAX * 1.4

FRONT_W, FRONT_H = 5.3, 4.9
FRONT_X = 3.0
FRONT_Y = -0.45
FRONT_X_MAX = CURVE_Y_MAX
FRONT_Y_MAX = CURVE_Y_MAX

# How the true curves are drawn, and how finely a parametric one is sampled.
CURVE_STROKE = 5.0
LANDSCAPE_STROKE = 3.5
PLOT_SAMPLES = 0.004

TITLE_EDGE_BUFF = 0.3

AXIS_CONFIG = {
    "color": INK,
    "stroke_width": 2.5,
    "include_ticks": False,
    "tip_length": 0.18,
    "tip_width": 0.18,
}


def action_axes(y):
    """Axes for one objective against the action, in the left column."""
    axes = Axes(
        x_range=[0.0, AXIS_X_MAX, 1.0],
        y_range=[0.0, AXIS_Y_MAX, 1.0],
        x_length=ACTION_W,
        y_length=ACTION_H,
        axis_config=AXIS_CONFIG,
    )
    axes.move_to([ACTION_X, y, 0])
    return axes


def action_plot(axes, label, func, domain, color, x_label=False):
    """That objective's labels and true curve, the labels in its own color.

    Only the lower plot is given the action label, since the two share an x
    axis; it goes at the right end, leaving the space under the axis for a
    scene's own labels.
    """
    axes.y_axis.set_color(color)
    text = Tex(label, font_size=AXIS_LABEL_SIZE, color=color)
    text.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
    labels = VGroup(text)
    if x_label:
        labels.add(action_label().next_to(axes.x_axis.get_end(), DOWN, buff=0.2))
    curve = axes.plot(func, x_range=domain, color=color, stroke_width=CURVE_STROKE)
    return VGroup(labels, curve)


def front_axes():
    """The objective space: metabolic cost across, comfort up."""
    axes = Axes(
        x_range=[0.0, FRONT_X_MAX, 1.0],
        y_range=[0.0, FRONT_Y_MAX, 1.0],
        x_length=FRONT_W,
        y_length=FRONT_H,
        axis_config=AXIS_CONFIG,
    )
    axes.move_to([FRONT_X, FRONT_Y, 0])
    axes.x_axis.set_color(COST_COLOR)
    axes.y_axis.set_color(COMFORT_COLOR)

    x_label = Tex("metabolic cost", font_size=AXIS_LABEL_SIZE, color=COST_COLOR)
    x_label.next_to(axes.x_axis, DOWN, buff=0.18)
    y_label = Tex("comfort", font_size=AXIS_LABEL_SIZE, color=COMFORT_COLOR)
    y_label.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
    return axes, VGroup(x_label, y_label)
