"""Synthetic validation: the three ablations' IGD+ curves, side by side under one legend.

Each panel is the figure one of `scripts/icra/final_noise.py`, `final_dim.py` and
`final_obj.py` writes, redrawn in manim. The panels are built one at a time in that
order carrying MO-HILBO alone, and then, a beat after the third, every panel's
NSGA-II is drawn at once, so the comparison lands across all three together rather
than panel by panel. The curves come from `final_dim`'s own `load`, `conditions` and
`shades`, so the video reads the same CSVs at the same query budget in the same
colors as the paper figures, and cannot drift from them.

The legend is `scripts/table.tex` as a key: one column per condition, light shade
first, the top two rows each method's ramp at that shade and the rows below what that
shade means in each panel.
"""

import sys
from dataclasses import dataclass
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
    DashedVMobject,
    FadeIn,
    LaggedStart,
    Line,
    ManimColor,
    MathTex,
    Polygon,
    Scene,
    Tex,
    VGroup,
    VMobject,
    Write,
)
from matplotlib.colors import to_hex

# manim loads this file by path rather than as a package, so the repo root is put on
# sys.path for the `video.style` and `scripts.icra` imports below.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.icra.final_dim import METHODS, conditions, load, shades  # noqa: E402
from video.style import AXIS_LABEL_SIZE, INK, SCENE_TITLE_SIZE  # noqa: E402

TITLE = "Synthetic Validation"
TITLE_EDGE_BUFF = 0.25


@dataclass(frozen=True)
class Ablation:
    """One panel: where its runs are, and what its conditions are called.

    Attributes:
        prefix: the condition subdirectory prefix, as `final_dim.conditions` takes it.
        dataset: the ablation directory of `<prefix><k>/` run directories.
        title: the panel title, naming the symbol the condition is swept in.
        symbol: that symbol alone, as the legend's row label.
        labels: the legend row's entries, light shade first, one per condition.
        ylim: the panel's y range, as its own script sets it.
    """

    prefix: str
    dataset: Path
    title: str
    symbol: str
    labels: tuple
    ylim: tuple


# Left to right, which is also the order they are revealed in. Each `ylim` is the one
# that ablation's own script passes to `final_dim.figure`, so the three panels are at
# different y scales and their ticks are what says so.
ABLATIONS = (
    Ablation(
        prefix="noise",
        dataset=ROOT / "scripts/output/final_exp/noise_ablation-100",
        title=r"noise $\sigma$",
        symbol=r"\sigma",
        labels=("0.01", "0.1", "0.2", "0.4"),
        ylim=(-0.05, 0.35),
    ),
    Ablation(
        prefix="dim",
        dataset=ROOT / "scripts/output/final_exp/dim-100-again",
        title=r"dimension $n$",
        symbol="n",
        labels=("1", "2", "3", "4"),
        ylim=(-0.05, 0.30),
    ),
    Ablation(
        prefix="objs",
        dataset=ROOT / "scripts/output/final_exp/obj-100",
        title=r"objectives $m$",
        symbol="m",
        # The run directories are objs1..objs4 and each holds that many measured
        # objectives, so `final_obj.py` labels the lines m=1..4; these are
        # `scripts/table.tex`'s numbers for the same four conditions.
        labels=("2", "3", "4", "5"),
        ylim=(-0.05, 0.50),
    ),
)

# Panel geometry, in manim scene units inside the 14.2 x 8.0 frame. The spacing is
# what keeps a panel's y tick labels clear of its neighbour's x axis; the row is
# centered on the frame after it is built, since the leftmost panel's y label sticks
# out where the rightmost one's x axis only reaches its tip.
# The two methods, in the order the scene reveals them.
HILBO, NSGA = METHODS

PANEL_X = (-4.7, 0.0, 4.7)
PANEL_Y = -1.15
PANEL_W, PANEL_H = 3.3, 3.3
TITLE_Y = 0.95

X_MAX = 105.0  # past the 100-query budget so a tick lands on 100, and for the tip
X_STEP = 25.0
Y_STEP = 0.1
NUMBER_SIZE = 22

CURVE_STROKE = 4.0
NUM_DASHES = 26
# Matches matplotlib's own band opacity, now that this scene is on white too.
BAND_OPACITY = 0.12

# Legend geometry. The block is built about the origin and moved as a whole, so the
# wide row labels do not pull it off center.
LEGEND_Y = 2.35
LEGEND_COL_W = 1.05
LEGEND_ROW_H = 0.33
LEGEND_LABEL_GAP = 0.25
LEGEND_SIZE = 26
SWATCH_W = 0.55
SWATCH_STROKE = 5.0
SWATCH_DASH = 0.075


def ramp_colors(ramp, n):
    """`n` manim colors from the light to the dark end of one method's matplotlib ramp."""
    return [ManimColor(to_hex(c)) for c in shades(ramp, n)]


def on_axes(axes, x, y, ylim):
    """The points `(x, y)` on `axes`, with `y` held inside the panel's own y range.

    matplotlib clips a curve and its fill to the axes; manim's `c2p` extrapolates past
    them, so a +/-1 std band leaving the panel would otherwise be drawn over its
    neighbour. It is a no-op for the means, which are inside every panel's `ylim`.
    """
    return [axes.c2p(a, b) for a, b in zip(x, np.clip(y, *ylim))]


def band(axes, x, low, high, color, ylim):
    """The +/-1 std band, as the polygon up `high` and back along `low`."""
    points = on_axes(axes, x, high, ylim) + on_axes(axes, x[::-1], low[::-1], ylim)
    polygon = Polygon(*points, stroke_width=0, fill_color=color, fill_opacity=BAND_OPACITY)
    return polygon.set_z_index(-1)  # behind the means, whichever is animated first


def curve(axes, x, y, color, ylim, dashed):
    """One mean curve as a polyline, dashed for NSGA-II as in the paper figures."""
    line = VMobject(stroke_color=color, stroke_width=CURVE_STROKE)
    line.set_points_as_corners(on_axes(axes, x, y, ylim))
    return DashedVMobject(line, num_dashes=NUM_DASHES, dashed_ratio=0.6) if dashed else line


class ValidationScene(Scene):
    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        runs = {a.prefix: conditions(a.dataset, a.prefix) for a in ABLATIONS}
        for a in ABLATIONS:
            if len(runs[a.prefix]) != len(a.labels):
                raise SystemExit(f"{len(runs[a.prefix])} {a.prefix}<k>/ runs under "
                                 f"{a.dataset} for {len(a.labels)} legend entries")

        panels = [self._panel(a, runs[a.prefix], x) for a, x in zip(ABLATIONS, PANEL_X)]
        row = VGroup(*(m for frame, methods in panels
                       for m in (frame, *(g for pair in methods.values() for g in pair))))
        row.shift(LEFT * row.get_center()[0])

        self.play(Write(title, run_time=1.0))
        self.play(FadeIn(self._legend(len(ABLATIONS[0].labels)), shift=DOWN * 0.2,
                         run_time=0.9))
        self.wait(0.3)

        for frame, methods in panels:
            self.play(Create(frame[0], run_time=0.7), Write(frame[1], run_time=0.7))
            self.play(*self._reveal(methods[HILBO], 1.4))
            self.wait(0.4)
        self.wait(1.0)

        self.play(*(anim for _, methods in panels
                    for anim in self._reveal(methods[NSGA], 1.2)))
        self.wait(1.5)

    @staticmethod
    def _reveal(curves, run_time):
        """One method's curves on one panel: the means swept in with a lag between
        them, the band fading up underneath."""
        bands, means = curves
        return (FadeIn(bands, run_time=run_time),
                LaggedStart(*(Create(m) for m in means), lag_ratio=0.18,
                            run_time=run_time))

    def _panel(self, ablation, runs, x):
        """One ablation's axes and labels, and its curves by method.

        Returns `(VGroup(axes, labels), {method: (bands, means)})`.
        """
        axes = Axes(
            x_range=[0.0, X_MAX, X_STEP],
            y_range=[ablation.ylim[0], ablation.ylim[1], Y_STEP],
            x_length=PANEL_W,
            y_length=PANEL_H,
            axis_config={
                "color": INK,
                "stroke_width": 2.5,
                "tip_length": 0.14,
                "tip_width": 0.14,
                "font_size": NUMBER_SIZE,
            },
            x_axis_config={
                "numbers_to_include": np.arange(X_STEP, X_MAX, X_STEP),
                "decimal_number_config": {"num_decimal_places": 0},
            },
            y_axis_config={
                "numbers_to_include": np.arange(0.0, ablation.ylim[1] + 1e-9, Y_STEP),
                "decimal_number_config": {"num_decimal_places": 1},
            },
        )
        axes.move_to([x, PANEL_Y, 0])

        title = Tex(ablation.title, font_size=AXIS_LABEL_SIZE, color=INK)
        title.move_to([x, TITLE_Y, 0])
        x_label = Tex("Queries", font_size=AXIS_LABEL_SIZE, color=INK)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Tex(r"IGD$^+$", font_size=AXIS_LABEL_SIZE, color=INK)
        y_label.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.2)

        methods = {m: self._curves(axes, ablation, runs, m) for m in METHODS}
        return VGroup(axes, VGroup(title, x_label, y_label)), methods

    def _curves(self, axes, ablation, runs, method):
        """One method's band and mean curve per condition, light shade first."""
        _, ramp, style = METHODS[method]
        bands, means = VGroup(), VGroup()
        for (_, run_dir), color in zip(runs, ramp_colors(ramp, len(runs))):
            evals, values = load(run_dir, method)
            mean, std = values.mean(axis=0), values.std(axis=0)
            bands.add(band(axes, evals, mean - std, mean + std, color, ablation.ylim))
            means.add(curve(axes, evals, mean, color, ablation.ylim, style == "--"))
        return bands, means

    def _legend(self, n):
        """The key: a method's ramp per row above what each shade means per panel.

        Built about the origin and moved as a whole, since the row labels are much
        wider than the columns and would otherwise pull the block off center.
        """
        columns = [(i - (n - 1) / 2) * LEGEND_COL_W for i in range(n)]
        label_x = columns[0] - LEGEND_COL_W / 2 - LEGEND_LABEL_GAP
        rows = [(METHODS[m][0], METHODS[m][1], METHODS[m][2]) for m in METHODS]

        cells = VGroup()
        for r, (name, ramp, style) in enumerate(rows):
            y = -r * LEGEND_ROW_H
            cells.add(Tex(name, font_size=LEGEND_SIZE, color=INK)
                      .move_to([label_x, y, 0], aligned_edge=RIGHT))
            for x, color in zip(columns, ramp_colors(ramp, n)):
                ends = ([x - SWATCH_W / 2, y, 0], [x + SWATCH_W / 2, y, 0])
                cells.add(DashedLine(*ends, dash_length=SWATCH_DASH, dashed_ratio=0.6,
                                     color=color, stroke_width=SWATCH_STROKE)
                          if style == "--" else
                          Line(*ends, color=color, stroke_width=SWATCH_STROKE))

        for r, ablation in enumerate(ABLATIONS, start=len(rows)):
            y = -r * LEGEND_ROW_H
            cells.add(MathTex(ablation.symbol, font_size=LEGEND_SIZE, color=INK)
                      .move_to([label_x, y, 0], aligned_edge=RIGHT))
            for x, text in zip(columns, ablation.labels):
                cells.add(MathTex(text, font_size=LEGEND_SIZE, color=INK)
                          .move_to([x, y, 0]))
        return cells.move_to([0.0, LEGEND_Y, 0])
