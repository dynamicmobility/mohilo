"""One subject's run, trial by trial: the inferred Pareto front and the actions on it.

The same thing `hilo/analysis/plot_fit_gif.py` writes as a gif, drawn on black.
The posteriors are the ones the run itself held — an `ExperimentDataset` stores
each trial's fit, so a frame is read back rather than refit — and the front is
ordered by that script's own `front_order`, so the two cannot disagree about
which points are non-dominated.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    UP,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Polygon,
    Tex,
    Scene,
    ThreeDAxes,
    Transform,
    VGroup,
    VMobject,
    Write,
    rgb_to_color,
)
from manim.utils.space_ops import rotation_matrix

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypolar as plr  # noqa: E402

from hilo.analysis.plot_fit_gif import front_order, padded_limits  # noqa: E402
from video.style import (  # noqa: E402
    AXIS_CONFIG,
    AXIS_LABEL_SIZE,
    BOX_FILL,
    BOX_STROKE,
    COMFORT_COLOR,
    COST_COLOR,
    INK,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
)

BOX_FACE_OPACITY = 0.12

DATASET = Path(__file__).resolve().parents[1] / "human_data" / "MB05" / "MB05.json"
# What a subject is called in the paper, keyed by the id their run is recorded
# under. The two numberings are unrelated.
SUBJECTS = {"MB02": "Subject 1", "MB04": "Subject 2", "MB05": "Subject 3"}

# The scan the front is read off. manim draws every point as its own vector
# object, so this is two powers below the gif's 2^12 and still reads as a cloud.
SCAN = 2**11
SEED = 95

ACTION_LABELS = ("Hip Flexion Scale", "Hip Extension Scale", "Delay")

# Objective space, on the left.
PARETO_W, PARETO_H = 5.0, 4.4
PARETO_X, PARETO_Y = -3.5, -0.7
PAD = 0.05
N_TICKS = 6
TICK_STEPS = [1, 2, 2.5, 5, 10]
TICK_DECIMALS = 3

# Action space, on the right.
ACTION_LEN = 3.2
ACTION_X, ACTION_Y = 3.9, -0.7
# matplotlib's default 3D view, which is what the gif is drawn at. `ELEVATION`
# lifts the camera above the x-y plane and `AZIMUTH` swings it around z.
ELEVATION = 30 * np.pi / 180
AZIMUTH = -60 * np.pi / 180

ACTION_LABEL_SIZE = 26
TICK_LABEL_SIZE = 20
LABEL_BUFF = 0.2
# How far a tick number is nudged off the box, along the line from the box's
# projected center through it — see `action_ticks`.
TICK_LABEL_PUSH = 0.22

CLOUD_RADIUS = 0.034
CLOUD_OPACITY = 0.45
FRONT_RADIUS = 0.055
FRONT_STROKE = 2.5
PATH_STROKE = 3.0

# An action is drawn as its own RGB, so a point's color says where in the action
# box it sits -- the same convention `scripts/icra/human_pareto.py` uses, and
# with no lift, matching its colors exactly.

STEP_TIME = 0.45
HOLD = 1.0


def fitted_trials(dataset):
    """The trials of a run carrying a posterior, less the run's last one.

    Trials chosen before the first fit have no `state_dict` and so nothing to
    draw. Stated here rather than at each call site so every scene reading a run
    shows the same trials.
    """
    return [record.trial for record in dataset.trials
            if record.state_dict is not None][:-1]


def final_front(path, scan=SCAN, seed=SEED):
    """One run's last fitted front: `(name, actions, bounds)`.

    `actions` is `(k, 3)`, ordered along the first objective as `front_order`
    leaves them, so walking the rows is walking the front from one objective's
    optimum to the other's. `bounds` is the run's own `(2, 3)` action box, which
    is the frame those actions are stated in.
    """
    dataset = plr.ExperimentDataset.load(path)
    mogp = dataset.get_model(fitted_trials(dataset)[-1])
    X = plr.sample_actions(mogp.action_bounds, scan, "sobol", seed)
    nd_idx = front_order(mogp.posterior_at(X)[0], mogp.posterior_at(X, raw=True)[0])

    return dataset.name, X[nd_idx], mogp.action_bounds


def load_run(path=DATASET, scan=SCAN, seed=SEED):
    """Every fitted trial of a run, as the arrays a frame is drawn from.

    Trials chosen before the first fit carry no posterior and are skipped, and
    the run's last one is dropped.

    Returns:
        `(name, trials, X, colors, frames)`, where `X` is the `(n, 3)` Sobol scan, and
        `frames` is one `(raw_mu, nd_idx)` per trial: the posterior mean in the
        objectives' own units, and the non-dominated rows of it, ordered along
        the first objective.
    """
    dataset = plr.ExperimentDataset.load(path)
    trials = fitted_trials(dataset)
    models = [dataset.get_model(trial) for trial in trials]

    X = plr.sample_actions(models[-1].action_bounds, scan, "sobol", seed)
    frames = []
    for mogp in models:
        # Dominance is read in maximization space, where both objectives are
        # larger-is-better; the same means in raw units are what is drawn.
        mu = mogp.posterior_at(X)[0]
        raw_mu = mogp.posterior_at(X, raw=True)[0]
        frames.append((raw_mu, front_order(mu, raw_mu)))

    colors = models[-1].objectives.xtransform(X)
    return (dataset.name, trials, X, np.clip(colors, 0.0, 1.0), frames)


def nice_ticks(low, high, n=N_TICKS):
    """`(values, decimals)` for round ticks inside `[low, high]`.

    The values come from matplotlib's own locator, which is what picks the ticks
    in the gif this scene mirrors. `decimals` is the fewest that still write every
    tick exactly: rounding 4.05 to one place would label two different ticks the
    same distance apart as 3.9 and 4.1.
    """
    values = MaxNLocator(nbins=n, steps=TICK_STEPS).tick_values(low, high)
    values = values[(values >= low) & (values <= high)]
    decimals = next(d for d in range(TICK_DECIMALS + 1)
                    if np.allclose(values, np.round(values, d)))

    return values, decimals


def pareto_axes(frames):
    """Objective space, on one axis box spanning every trial's posterior."""
    low, high = padded_limits(np.vstack([raw_mu for raw_mu, _ in frames]), PAD)
    x_ticks, x_decimals = nice_ticks(low[0], high[0])
    y_ticks, y_decimals = nice_ticks(low[1], high[1])
    axes = Axes(
        x_range=[low[0], high[0], x_ticks[1] - x_ticks[0]],
        y_range=[low[1], high[1], y_ticks[1] - y_ticks[0]],
        x_length=PARETO_W,
        y_length=PARETO_H,
        axis_config={**AXIS_CONFIG, "include_ticks": True},
    ).move_to([PARETO_X, PARETO_Y, 0])
    axes.x_axis.set_color(COST_COLOR)
    axes.y_axis.set_color(COMFORT_COLOR)
    # Both objectives are measured quantities here, unlike the synthetic scenes',
    # so the ticks carry numbers.
    axes.x_axis.add_numbers(x_ticks, font_size=TICK_LABEL_SIZE,
                            num_decimal_places=x_decimals, color=COST_COLOR)
    axes.y_axis.add_numbers(y_ticks, font_size=TICK_LABEL_SIZE,
                            num_decimal_places=y_decimals, color=COMFORT_COLOR)

    x_label = Tex("metabolic cost", font_size=AXIS_LABEL_SIZE, color=COST_COLOR)
    x_label.next_to(axes.x_axis, DOWN, buff=0.18)
    y_label = Tex("comfort", font_size=AXIS_LABEL_SIZE, color=COMFORT_COLOR)
    y_label.rotate(np.pi / 2).next_to(axes.y_axis, LEFT, buff=0.18)
    return axes, VGroup(x_label, y_label)


def pareto_frame(axes, raw_mu, nd_idx, colors):
    """One trial's objective space: the scanned cloud, and the front over it.

    The cloud is every scanned point, in index order, non-dominated ones
    included: a trial is animated into the next with one `Transform`, which
    pairs the two groups submobject by submobject, so a dot only moves to where
    that same scan point went if the order holds. Drawing the dominated subset
    alone would change which point each dot is from trial to trial and send the
    whole cloud churning. The front's own markers are drawn over their dots.
    """
    cloud = VGroup(*(
        Dot(axes.c2p(*point), radius=CLOUD_RADIUS,
            color=rgb_to_color(color), fill_opacity=CLOUD_OPACITY)
        for point, color in zip(raw_mu, colors)
    ))
    line = VMobject(color=INK, stroke_width=FRONT_STROKE)
    line.set_points_as_corners([axes.c2p(*raw_mu[i]) for i in nd_idx])
    markers = VGroup(*(
        Dot(axes.c2p(*raw_mu[i]), radius=FRONT_RADIUS,
            color=rgb_to_color(colors[i]), stroke_color=INK, stroke_width=1.5)
        for i in nd_idx
    ))
    return cloud, line, markers


def action_axes(X, length=ACTION_LEN):
    """The 3D action box, spanning the scan, in its own unrotated frame.

    manim's own 3D camera belongs to a `ThreeDScene`, and this scene is flat.
    The view is a fixed parallel projection either way, so the axes and
    everything plotted on them are built here and rotated by `orient`: `c2p` is
    called in this frame and the rotation carries the result into the view.
    """
    low, high = X.min(axis=0), X.max(axis=0)
    axes = ThreeDAxes(
        x_range=[low[0], high[0], high[0] - low[0]],
        y_range=[low[1], high[1], high[1] - low[1]],
        z_range=[low[2], high[2], high[2] - low[2]],
        x_length=length,
        y_length=length,
        z_length=length,
        axis_config=AXIS_CONFIG,
    )
    return axes


def action_box_faces(axes, opacity=BOX_FACE_OPACITY):
    """The three faces of the action box meeting at its low corner, as translucent panels.

    Three bare axis lines converging at a point read as flat; matplotlib's own
    3D panes are what give a box its depth, so this draws the same cue: the
    floor and the two walls behind it, filled in the same box color every other
    scene's panels use. Built in `axes`'s own unrotated frame, like every other
    action-space mobject, so it takes the same `orient` the points on it do.
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = (
        axes.x_range[:2], axes.y_range[:2], axes.z_range[:2])
    corners = (
        ((x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo)),
        ((x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_lo, z_hi), (x_lo, y_lo, z_hi)),
        ((x_lo, y_lo, z_lo), (x_lo, y_hi, z_lo), (x_lo, y_hi, z_hi), (x_lo, y_lo, z_hi)),
    )
    faces = VGroup(*(
        Polygon(*(axes.c2p(*corner) for corner in face), color=BOX_STROKE,
                fill_color=BOX_FILL, fill_opacity=opacity, stroke_width=1.5)
        for face in corners
    ))
    faces.set_z_index(-1)
    return faces


# The three turns that carry the action frame into the drawn view. The first
# stands the z axis up — manim draws z out of the screen and a 3D plot draws it
# up — which leaves the camera on the -y axis, an azimuth of -90 degrees. The
# second makes up the rest of `AZIMUTH`, and the third lifts the camera by
# `ELEVATION`.
TURNS = ((-np.pi / 2, RIGHT), (-(AZIMUTH + np.pi / 2), UP), (ELEVATION, RIGHT))


def orient(mob, center=(ACTION_X, ACTION_Y)):
    """Rotate something built in the action frame into the drawn view, and place it.

    Every rotation is about the origin, which is where `ThreeDAxes` puts the
    action box's own center. Rotating about each mobject's own center instead
    would turn a path and the axes it lies on by different amounts.
    """
    for angle, axis in TURNS:
        mob.rotate(angle, axis, about_point=ORIGIN)
    return mob.shift([center[0], center[1], 0])


def project(points, center=(ACTION_X, ACTION_Y)):
    """The same view, applied to bare `(n, 3)` coordinates rather than to a mobject.

    A marker must be placed by this rather than rotated by `orient`: a `Dot` is a
    circle in the drawing plane, so turning it out of that plane squashes it into
    an ellipse seen edge-on. Lines are unaffected, since a rotated corner is
    still the projection of that corner.
    """
    points = np.asarray(points, dtype=float)
    for angle, axis in TURNS:
        points = points @ rotation_matrix(angle, axis).T

    return points + np.array([center[0], center[1], 0.0])


def action_labels(axes):
    """The three action names, each beyond the end of its already-rotated axis.

    The offset is along the axis's own drawn direction rather than a fixed one,
    which is what keeps a label off the box: under this view the y axis points
    up and to the right, so a label placed below its tip would land inside the
    plot.
    """
    labels = VGroup()
    for axis, name in zip((axes.x_axis, axes.y_axis, axes.z_axis), ACTION_LABELS):
        span = axis.get_end() - axis.get_start()
        text = Tex(name, font_size=ACTION_LABEL_SIZE, color=INK)
        labels.add(text.next_to(axis.get_end(), span / np.linalg.norm(span),
                                buff=LABEL_BUFF))
    return labels


def action_markers(axes, X, nd_idx, colors, center=(ACTION_X, ACTION_Y)):
    """One trial's non-dominated actions, as points in the action box.

    Placed by `project` rather than built on `axes` and rotated by `orient`,
    which is what keeps a marker a circle instead of an ellipse seen edge-on —
    the same reason `pareto_sets.py`'s own markers are placed this way.
    """
    points = project([axes.c2p(*X[i]) for i in nd_idx], center=center)
    return VGroup(*(
        Dot(point, radius=FRONT_RADIUS, color=rgb_to_color(colors[i]),
            stroke_color=INK, stroke_width=1.5)
        for point, i in zip(points, nd_idx)
    ))


def action_ticks(axes, center=(ACTION_X, ACTION_Y), n=3, font_size=TICK_LABEL_SIZE):
    """Numeric tick labels for the action box, one small cluster per axis.

    Placed by `project`, exactly like `action_markers`, for the reason
    `action_labels` states: text built and then rotated by `orient` would tilt
    edge-on instead of staying upright. Each label is nudged along the line from
    the box's own projected center through its tick, which is what fans the
    three axes' numbers out and away from the lines rather than stacking them
    on the edges — the same radial arrangement matplotlib's own 3D ticks fall
    into under a rotated view.
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = (
        axes.x_range[:2], axes.y_range[:2], axes.z_range[:2])
    box_center = project(
        [axes.c2p((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2)], center)[0]

    labels = VGroup()
    for dim, (lo, hi) in enumerate(((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))):
        ticks, decimals = nice_ticks(lo, hi, n=n)
        base = [x_lo, y_lo, z_lo]
        for t in ticks:
            point = list(base)
            point[dim] = t
            proj = project([axes.c2p(*point)], center)[0]
            direction = proj - box_center
            offset = TICK_LABEL_PUSH * direction / max(np.linalg.norm(direction[:2]), 1e-6)
            text = Tex(f"{t:.{decimals}f}", font_size=font_size, color=INK)
            text.move_to(proj + offset)
            labels.add(text)
    return labels


def action_path(axes, X, nd_idx):
    """The line joining those actions, in the order the front puts them."""
    line = VMobject(color=INK, stroke_width=PATH_STROKE)
    line.set_points_as_corners([axes.c2p(*X[i]) for i in nd_idx])
    line.set_z_index(-1)

    return line


class FrontScene(Scene):
    """MB05's run: every fitted trial's front, in objective space and in action space."""

    def construct(self):
        name, trials, X, colors, frames = load_run()

        title = Tex(f"{SUBJECTS[name]} Optimization",
                    font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        p_axes, p_labels = pareto_axes(frames)
        a_axes = action_axes(X)

        raw_mu, nd_idx = frames[0]
        cloud, line, markers = pareto_frame(p_axes, raw_mu, nd_idx, colors)

        # Everything in the action box takes the same rotation, so a point's
        # place in it is fixed before the view is chosen. `a_ref` keeps the
        # unrotated axes, since every later frame's `c2p` is read off them.
        a_ref = a_axes.copy()
        a_faces = orient(action_box_faces(a_ref))
        orient(a_axes)
        a_labels = action_labels(a_axes)
        a_ticks = action_ticks(a_ref)
        path_markers = action_markers(a_ref, X, nd_idx, colors)

        counter = self.counter(trials[0], p_axes)

        self.play(Write(title, run_time=1.0))
        self.play(Create(p_axes), FadeIn(a_faces), Create(a_axes),
                  FadeIn(p_labels, a_labels, a_ticks))
        self.play(FadeIn(cloud, line, markers, path_markers, counter))
        self.wait(HOLD)

        for trial, (raw_mu, nd_idx) in zip(trials[1:], frames[1:]):
            new_cloud, new_line, new_markers = pareto_frame(p_axes, raw_mu, nd_idx, colors)
            self.play(
                Transform(cloud, new_cloud),
                Transform(line, new_line),
                Transform(markers, new_markers),
                Transform(path_markers, action_markers(a_ref, X, nd_idx, colors)),
                Transform(counter, self.counter(trial, p_axes)),
                run_time=STEP_TIME,
            )

        # The action-space points are joined only once, on the run's last fit:
        # which actions are on the front changes every trial, so a line drawn
        # along the way says more about the ordering than about the front.
        path = orient(action_path(a_ref, X, nd_idx))
        self.play(Create(path))
        self.wait(HOLD)
        self.play(FadeOut(VGroup(title, p_axes, p_labels, a_axes, a_faces, a_labels,
                                 a_ticks, cloud, line, markers, path, path_markers, counter)))

    def counter(self, trial, axes):
        """The trial number, under the objective-space panel.

        A dataset records trials from zero and a run is counted from one, so the
        drawn number is one above the record's.
        """
        text = Tex(f"Trial {trial + 1}", font_size=AXIS_LABEL_SIZE, color=INK)
        return text.next_to(axes, UP, buff=0.2)
