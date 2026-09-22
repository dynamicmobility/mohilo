"""Three subjects side by side: each one's final front, over its final set.

One column per subject, front on top and set below it, so the two rows read as
"what they achieved" against "how they achieved it" without the trial-by-trial
animation `FrontScene` runs for one subject at a time.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    UP,
    Axes,
    Create,
    Dot,
    FadeOut,
    Scene,
    Tex,
    VGroup,
    VMobject,
    Write,
    rgb_to_color,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypolar as plr  # noqa: E402

from hilo.analysis.plot_fit_gif import front_order, padded_limits  # noqa: E402
from video.front import (  # noqa: E402
    ACTION_LABELS,
    FRONT_RADIUS,
    FRONT_STROKE,
    SUBJECTS,
    action_axes,
    action_box_faces,
    action_markers,
    action_path,
    action_ticks,
    fitted_trials,
    nice_ticks,
    orient,
)
from video.style import (  # noqa: E402
    AXIS_CONFIG,
    BG,
    COMFORT_COLOR,
    COST_COLOR,
    INK,
    TITLE_EDGE_BUFF,
)

LABEL_HALO = 6.0

HUMAN_DATA = Path(__file__).resolve().parents[1] / "human_data"
RUNS = ("MB02", "MB04", "MB05")
TITLE = "Fronts and Sets Across Subjects"

SCAN = 2**11
SEED = 95

GRID_TITLE_SIZE = 38
TICK_LABEL_SIZE = 18
FRONT_AXIS_LABEL_SIZE = 20
NAME_LABEL_SIZE = 26
ACTION_LABEL_SIZE = 18
ACTION_TICK_SIZE = 14
N_TICKS = 3

# Column centers, spanning manim's 14.2-wide frame, and the two rows' heights.
COL_X = (-4.55, 0.0, 4.55)
FRONT_Y = 1.35
FRONT_W, FRONT_H = 3.8, 2.1
SET_Y = -2.5
SET_LEN = 2.0

# `scripts/icra/human_pareto.py`'s own dominated-point alpha (`dominated_alpha`
# in its `plot_pareto` call); the radius is smaller than `front.py`'s own
# `CLOUD_RADIUS` because a grid cell is a fraction of that scene's panel.
CLOUD_RADIUS = 0.03
CLOUD_OPACITY = 0.12


def subject_front_and_set(name, scan=SCAN, seed=SEED):
    """One subject's last fitted trial: the whole scan, and which of it is on the front.

    Returns `(name, X, raw_mu, nd_idx, colors)`: `X` and `raw_mu` are `(n, ...)`,
    the full Sobol scan and its posterior mean in the objectives' own units;
    `nd_idx` are the front's indices into both, ordered along the first
    objective by `front_order`; `colors` are `(n, 3)`, one action-derived RGB
    per scanned point — the same convention `scripts/icra/human_pareto.py`
    colors a run by, with no lift, matching its colors exactly. Returning the
    whole scan rather than just the front is what
    lets a caller draw every point `human_pareto.py` does, dominated ones
    included, not only the ones that survive to the front.
    """
    dataset = plr.ExperimentDataset.load(HUMAN_DATA / name / f"{name}.json")
    mogp = dataset.get_model(fitted_trials(dataset)[-1])

    X = plr.sample_actions(mogp.action_bounds, scan, "sobol", seed)
    mu = mogp.posterior_at(X)[0]
    raw_mu = mogp.posterior_at(X, raw=True)[0]
    nd_idx = front_order(mu, raw_mu)

    colors = np.clip(mogp.objectives.xtransform(X), 0.0, 1.0)

    return (SUBJECTS[name], X, raw_mu, nd_idx, colors)


def small_front_axes(center, raw_mu, width=FRONT_W, height=FRONT_H, pad=0.08):
    """A front-space axes box, sized for one grid cell, with numbered ticks.

    Sized off the whole scan `raw_mu`, not just the front, since the cloud is
    drawn too and the box must hold all of it. `width`, `height` and `pad`
    default to this module's own grid-cell geometry; a caller with a
    differently sized panel, such as `subjects_validation.py`'s single-row
    layout, passes its own.
    """
    low, high = padded_limits(raw_mu, pad)
    x_ticks, x_decimals = nice_ticks(low[0], high[0], n=N_TICKS)
    y_ticks, y_decimals = nice_ticks(low[1], high[1], n=N_TICKS)
    axes = Axes(
        x_range=[low[0], high[0], x_ticks[1] - x_ticks[0]],
        y_range=[low[1], high[1], y_ticks[1] - y_ticks[0]],
        x_length=width,
        y_length=height,
        axis_config={**AXIS_CONFIG, "include_ticks": True},
    ).move_to([center[0], center[1], 0])
    axes.x_axis.set_color(COST_COLOR)
    axes.y_axis.set_color(COMFORT_COLOR)
    axes.x_axis.add_numbers(x_ticks, font_size=TICK_LABEL_SIZE,
                            num_decimal_places=x_decimals, color=COST_COLOR)
    axes.y_axis.add_numbers(y_ticks, font_size=TICK_LABEL_SIZE,
                            num_decimal_places=y_decimals, color=COMFORT_COLOR)

    x_label = Tex("metabolic cost", font_size=FRONT_AXIS_LABEL_SIZE, color=COST_COLOR)
    x_label.next_to(axes.x_axis, DOWN, buff=0.12)
    y_label = Tex("comfort", font_size=FRONT_AXIS_LABEL_SIZE, color=COMFORT_COLOR)
    y_label.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.12)
    return axes, VGroup(x_label, y_label)


def small_action_labels(axes, font_size=ACTION_LABEL_SIZE):
    """The three action names beyond their already-rotated axes, at grid-cell size.

    Copies `front.py`'s `action_labels` at a smaller font — this grid gives each
    column a fraction of the room `FrontScene`'s own action panel has.
    """
    labels = VGroup()
    for axis, name in zip((axes.x_axis, axes.y_axis, axes.z_axis), ACTION_LABELS):
        span = axis.get_end() - axis.get_start()
        text = Tex(name, font_size=font_size, color=INK)
        labels.add(text.next_to(axis.get_end(), span / np.linalg.norm(span), buff=0.12))
    return labels


def front_cloud(axes, raw_mu, nd_idx, colors):
    """One subject's whole scan in objective space: every point, colored by its
    action, dominated ones faded, the front outlined over them.

    Mirrors `scripts/icra/human_pareto.py`'s `plot_pareto(show_dominated=True,
    dominated_alpha=0.1, outline_nondominated=1)`: the same coloring, and the
    same distinction between a point that only shows where the scan reached and
    one that is actually optimal.
    """
    cloud = VGroup(*(
        Dot(axes.c2p(*point), radius=CLOUD_RADIUS, color=rgb_to_color(color),
            fill_opacity=CLOUD_OPACITY)
        for point, color in zip(raw_mu, colors)
    ))
    line = VMobject(color=INK, stroke_width=FRONT_STROKE)
    line.set_points_as_corners([axes.c2p(*raw_mu[i]) for i in nd_idx])
    markers = VGroup(*(
        Dot(axes.c2p(*raw_mu[i]), radius=FRONT_RADIUS, color=rgb_to_color(colors[i]),
            stroke_color=INK, stroke_width=1.5)
        for i in nd_idx
    ))
    return cloud, line, markers


class SubjectsGridScene(Scene):
    """MB02, MB04 and MB05's final fronts, each over its own final set."""

    def construct(self):
        title = Tex(TITLE, font_size=GRID_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        self.play(Write(title, run_time=1.0))

        columns = VGroup()
        for run, x in zip(RUNS, COL_X):
            name, X, raw_mu, nd_idx, colors = subject_front_and_set(run)
            set_actions = X[nd_idx]

            f_axes, f_labels = small_front_axes((x, FRONT_Y), raw_mu)
            cloud, line, markers = front_cloud(f_axes, raw_mu, nd_idx, colors)
            name_label = Tex(name, font_size=NAME_LABEL_SIZE, color=INK)
            name_label.next_to(f_axes, UP, buff=0.2)

            # `a_ref` keeps the unrotated axes, since every marker, tick, face
            # and path below are placed by its `c2p` before `orient` turns the
            # drawn axes into the view.
            a_axes = action_axes(X, length=SET_LEN)
            a_ref = a_axes.copy()
            a_faces = orient(action_box_faces(a_ref), center=(x, SET_Y))
            orient(a_axes, center=(x, SET_Y))
            a_labels = small_action_labels(a_axes)
            # The set crowds the axis tips, which is where the names belong.
            # A background stroke in the scene color keeps a name readable
            # over the points it lands on, rather than moving it off its axis.
            a_labels.set_stroke(BG, width=LABEL_HALO, background=True)
            a_labels.set_z_index(1)
            a_ticks = action_ticks(a_ref, center=(x, SET_Y), n=N_TICKS, font_size=ACTION_TICK_SIZE)
            a_ticks.set_stroke(BG, width=LABEL_HALO, background=True)
            a_ticks.set_z_index(1)
            # `action_markers` places each `Dot` by `project` rather than
            # rotating one built on `a_ref`, which is what keeps it a circle
            # instead of an ellipse seen edge-on.
            a_markers = action_markers(a_ref, set_actions, range(len(set_actions)),
                                       colors[nd_idx], center=(x, SET_Y))
            a_path = orient(action_path(a_ref, set_actions, range(len(set_actions))),
                            center=(x, SET_Y))

            columns.add(VGroup(f_axes, f_labels, cloud, line, markers, name_label,
                               a_faces, a_axes, a_labels, a_ticks, a_markers, a_path))

        self.play(Create(columns))
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, columns)))
