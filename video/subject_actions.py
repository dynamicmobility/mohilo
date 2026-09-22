"""Every subject's inferred Pareto set, animated in manim.

The manim counterpart of `hilo/analysis/plot_actions_video.py`: the same data
(`scripts/icra/subject_actions.py`'s `pareto_actions` and `SUBJECT_COLORS`, and
`scripts/icra/subjects_pareto.py`'s `SUBJECTS` display names), drawn as an
actual `ThreeDScene` at the same `ELEV`/`AZIM` matplotlib's `mplot3d` uses, so
the axes read as the same boxed, gridded panes matplotlib draws rather than
this package's other, hand-projected 3D scenes (`video/front.py`,
`video/pareto_sets.py`).
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    RIGHT,
    UP,
    Create,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    Polygon,
    Tex,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypolar as plr  # noqa: E402

from scripts.icra.human_pareto import ACTION_LABELS  # noqa: E402
from scripts.icra.subject_actions import SUBJECT_COLORS, pareto_actions  # noqa: E402
from scripts.icra.subjects_pareto import SUBJECTS  # noqa: E402
from video.front import nice_ticks  # noqa: E402 (a numeric utility, not a style choice)
from video.style import BG, INK, SCENE_TITLE_SIZE, TITLE_EDGE_BUFF  # noqa: E402

HUMAN_DATA = Path(__file__).resolve().parents[1] / "human_data"
TITLE = "Subject Comparison"

TRIAL = -1        # the optimization trial whose GP each front is read from
SCAN = 2**14      # Sobol points the front is read off, matching the mp4 exactly
SEED = 95

# `scripts/icra/subject_actions.py`'s own 3D view. mplot3d's `elev` is degrees
# above the x-y plane, which is manim's polar angle measured the other way
# from the z-axis (`phi = 90 - elev`); `azim` and manim's own `theta` share the
# same convention -- rotation about z, from the x-axis -- directly.
ELEV, AZIM = 25, -60
PHI = (90 - ELEV) * DEGREES
THETA = AZIM * DEGREES

BOX_LEN = 5.4
N_TICKS = 4

# matplotlib's own `mplot3d` defaults: a pane per axis at ~0.9-0.95 gray, 0.5
# opacity over the white figure, `grid.color` #b0b0b0 for both the grid lines
# and the box edges (mplot3d draws no separate, darker spine).
PANE_COLOR = ManimColor("#F0F0F0")
PANE_OPACITY = 0.5
LINE_COLOR = ManimColor("#B0B0B0")
LINE_WIDTH = 1.5

TICK_LABEL_SIZE = 20
AXIS_LABEL_SIZE = 24
TICK_PUSH = 0.35
AXIS_LABEL_PUSH = 0.85

MARKER_RADIUS = 0.07

LEGEND_X, LEGEND_Y = -6.3, 3.0
LEGEND_BUFF = 0.3
LEGEND_DOT_RADIUS = 0.09

STEP_TIME = 1.2
HOLD = 3.0


def load_fronts(subjects=SUBJECTS, data_dir=HUMAN_DATA, trial=TRIAL, scan=SCAN, seed=SEED):
    """`(names, fronts, box)`: each subject's display name, its `(k, 3)` raw
    Pareto actions, and the `(2, 3)` union of every subject's action box.

    Mirrors `plot_actions_video.make_video`'s own bounds/fronts computation
    exactly, so the two draw the same points in the same frame.
    """
    datasets = [plr.ExperimentDataset.load(data_dir / s / f"{s}.json") for s in subjects]
    boxes = [plr.as_bounds(d.get_model(trial).action_bounds) for d in datasets]
    box = np.array([np.min([b[0] for b in boxes], axis=0),
                    np.max([b[1] for b in boxes], axis=0)])
    fronts = [pareto_actions(d, trial, scan, seed) for d in datasets]

    return [subjects[s] for s in subjects], fronts, box


def grid_pane(axes, corners, u_ticks, u_line, v_ticks, v_line):
    """One matplotlib-style pane: a light gray fill plus its own tick grid.

    `corners` are the four data-space corners of the rectangle, in order.
    `u_line(t)`/`v_line(t)` each return the two data-space endpoints of the
    grid line at tick `t` along that pane's two in-plane axes.
    """
    face = Polygon(*(axes.c2p(*c) for c in corners), fill_color=PANE_COLOR,
                   fill_opacity=PANE_OPACITY, stroke_color=LINE_COLOR,
                   stroke_width=LINE_WIDTH)
    lines = VGroup(*(
        Line(axes.c2p(*a), axes.c2p(*b), color=LINE_COLOR, stroke_width=LINE_WIDTH)
        for ticks, line in ((u_ticks, u_line), (v_ticks, v_line))
        for t in ticks
        for a, b in [line(t)]
    ))
    return VGroup(face, lines)


def action_box(axes, box, x_ticks, y_ticks, z_ticks):
    """The three matplotlib-visible panes at this view: the floor and the two
    back walls, whose shared boundary is what reads as the whole wireframe box.
    """
    (x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi) = box

    floor = grid_pane(
        axes, [(x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo)],
        x_ticks, lambda t: ((t, y_lo, z_lo), (t, y_hi, z_lo)),
        y_ticks, lambda t: ((x_lo, t, z_lo), (x_hi, t, z_lo)),
    )
    back = grid_pane(
        axes, [(x_lo, y_hi, z_lo), (x_hi, y_hi, z_lo), (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi)],
        x_ticks, lambda t: ((t, y_hi, z_lo), (t, y_hi, z_hi)),
        z_ticks, lambda t: ((x_lo, y_hi, t), (x_hi, y_hi, t)),
    )
    side = grid_pane(
        axes, [(x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_hi, y_hi, z_hi), (x_hi, y_lo, z_hi)],
        y_ticks, lambda t: ((x_hi, t, z_lo), (x_hi, t, z_hi)),
        z_ticks, lambda t: ((x_hi, y_lo, t), (x_hi, y_hi, t)),
    )
    return VGroup(floor, back, side)


def axis_ticks_and_label(axes, ticks, decimals, point_at, push_dir, label, label_t):
    """One axis's tick numbers plus its name, pushed out along `push_dir`.

    `point_at(t)` is that axis's data-space point at tick `t`, on the box edge
    matplotlib draws the axis and its ticks along. `push_dir` is a unit vector
    in the scene's own coordinates -- the outward direction off that edge --
    so the same offset works regardless of which data range the axis spans.
    """
    ticks_group = VGroup(*(
        Tex(f"{t:.{decimals}f}", font_size=TICK_LABEL_SIZE, color=INK).move_to(
            axes.c2p(*point_at(t)) + TICK_PUSH * push_dir)
        for t in ticks
    ))
    label_tex = Tex(label, font_size=AXIS_LABEL_SIZE, color=INK).move_to(
        axes.c2p(*point_at(label_t)) + AXIS_LABEL_PUSH * push_dir)

    return ticks_group, label_tex


def subject_markers(axes, actions, color):
    """One subject's Pareto actions as same-colored spheres in the action box.

    Not joined -- `scripts/icra/subject_actions.py`'s own `plot_subject`
    scatters its front and leaves the connecting line commented out, so a
    subject's set here is drawn the same way.
    """
    return VGroup(*(
        Dot3D(axes.c2p(*action), radius=MARKER_RADIUS, color=color)
        for action in actions
    ))


def legend(names, colors):
    """A colored dot per subject beside its name, as one stacked block."""
    rows = VGroup()
    for name, color in zip(names, colors):
        dot = Dot3D(radius=LEGEND_DOT_RADIUS, color=color, resolution=(8, 8))
        text = Tex(name, font_size=AXIS_LABEL_SIZE, color=color)
        text.next_to(dot, RIGHT, buff=0.18)
        rows.add(VGroup(dot, text))
    rows.arrange(DOWN, buff=LEGEND_BUFF, aligned_edge=np.array([-1.0, 0.0, 0.0]))

    return rows.move_to([LEGEND_X, LEGEND_Y, 0])


class SubjectActionsScene(ThreeDScene):
    """Every subject's inferred Pareto set, revealed one after another in one
    matplotlib-style 3D action box."""

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)

        names, fronts, box = load_fronts()
        colors = SUBJECT_COLORS[:len(names)]

        (x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi) = box
        x_ticks, x_dec = nice_ticks(x_lo, x_hi, n=N_TICKS)
        y_ticks, y_dec = nice_ticks(y_lo, y_hi, n=N_TICKS)
        z_ticks, z_dec = nice_ticks(z_lo, z_hi, n=N_TICKS)

        axes = ThreeDAxes(
            x_range=[x_lo, x_hi, x_hi - x_lo], y_range=[y_lo, y_hi, y_hi - y_lo],
            z_range=[z_lo, z_hi, z_hi - z_lo], x_length=BOX_LEN, y_length=BOX_LEN,
            z_length=BOX_LEN,
        )
        axes.set_opacity(0)   # only its `c2p` is used; the drawn box is `action_box`

        panes = action_box(axes, box, x_ticks, y_ticks, z_ticks)

        # World-space outward directions, read off the (already rotated) axes
        # rather than assumed, so the push is correct whichever way the box
        # ends up oriented on screen.
        x_dir = axes.c2p(x_hi, y_lo, z_lo) - axes.c2p(x_lo, y_lo, z_lo)
        y_dir = axes.c2p(x_lo, y_hi, z_lo) - axes.c2p(x_lo, y_lo, z_lo)
        z_dir = axes.c2p(x_lo, y_lo, z_hi) - axes.c2p(x_lo, y_lo, z_lo)
        x_dir, y_dir, z_dir = (v / np.linalg.norm(v) for v in (x_dir, y_dir, z_dir))

        x_tick_labels, x_label = axis_ticks_and_label(
            axes, x_ticks, x_dec, lambda t: (t, y_lo, z_lo), -y_dir - 0.4 * z_dir,
            ACTION_LABELS[0], (x_lo + x_hi) / 2)
        y_tick_labels, y_label = axis_ticks_and_label(
            axes, y_ticks, y_dec, lambda t: (x_hi, t, z_lo), x_dir - 0.4 * z_dir,
            ACTION_LABELS[1], (y_lo + y_hi) / 2)
        z_tick_labels, z_label = axis_ticks_and_label(
            axes, z_ticks, z_dec, lambda t: (x_lo, y_hi, t), x_dir,
            ACTION_LABELS[2], (z_lo + z_hi) / 2)

        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        key = legend(names, colors)

        text_labels = VGroup(x_tick_labels, y_tick_labels, z_tick_labels,
                             x_label, y_label, z_label)
        self.add_fixed_orientation_mobjects(*text_labels)
        self.add_fixed_in_frame_mobjects(title, key)
        self.remove(title, key, *text_labels)

        self.play(Write(title, run_time=1.0))
        self.play(Create(panes), FadeIn(text_labels))

        drawn = VGroup()
        for front, color, row in zip(fronts, colors, key):
            markers = subject_markers(axes, front, color)
            drawn.add(markers)
            self.play(FadeIn(row), FadeIn(markers), run_time=STEP_TIME)

        self.wait(HOLD)
        self.play(FadeOut(VGroup(title, panes, text_labels, key, drawn)))
