"""Shared palette, geometry and manim config for every scene in this package.

Importing this module sets the render background, the LaTeX template and the
media directory, so every scene looks the same without repeating the
configuration.
"""

from pathlib import Path

from manim import (
    DOWN,
    Arrow,
    ManimColor,
    MathTex,
    RoundedRectangle,
    Tex,
    TexTemplate,
    VGroup,
    config,
)

# Palette: light ink on black, one color per objective and one accent for the optimum.
BG = ManimColor("#000000")
INK = ManimColor("#F2F2F2")
BOX_FILL = ManimColor("#12171D")
BOX_STROKE = ManimColor("#8AA0B4")
ARROW_COLOR = ManimColor("#C7D3DE")
COST_COLOR = ManimColor("#4CC9F0")
COMFORT_COLOR = ManimColor("#F25F5C")
STAR_COLOR = ManimColor("#FFB703")
SAMPLE_COLOR = ManimColor("#F2F2F2")
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
