"""Three subjects' fronts, then their held-out validation points.

Draws the same panel `subjects_grid.py`'s top row does -- one subject's
inferred front per column -- and then replays `scripts/icra/validation_pareto.py`
on top of it, one source at a time: every subject's stars (the GP's
prediction) first, then every subject's squares (what was measured) together
with the line joining each to its own star, and only then the same two beats
for the anti-Pareto actions.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    Scene,
    Square,
    Star,
    Tex,
    VGroup,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypolar as plr  # noqa: E402

from hilo.analysis.plot_fit_gif import padded_limits  # noqa: E402
from scripts.icra.validation_pareto import (  # noqa: E402
    MATCH_TOL,
    SOURCE_COLORS,
    SOURCE_LABELS,
    front_order,
    measured_at,
)
from video.front import SUBJECTS  # noqa: E402
from video.style import AXIS_LABEL_SIZE, INK, SCENE_TITLE_SIZE, TITLE_EDGE_BUFF  # noqa: E402
from video.subjects_grid import (  # noqa: E402
    HUMAN_DATA,
    NAME_LABEL_SIZE,
    RUNS,
    front_cloud,
    small_front_axes,
)

TITLE = "Validation Against Held-Out Actions"

TRIAL = -1        # the optimization trial whose GP is drawn, and predicts
SCAN = 2**11       # Sobol points the front is read off
SEED = 95

COL_X = (-4.7, 0.0, 4.7)
FRONT_Y = -0.4
FRONT_W, FRONT_H = 3.9, 3.6
PAD = 0.08

STAR_SIZE = 0.17
SQUARE_SIZE = 0.13
MARKER_STROKE = 1.5
LINE_STROKE = 3.0

LEGEND_Y = 3.15
LEGEND_GAP = 0.55
LEGEND_SIZE = 22

HOLD = 1.2


def subject_panel_data(run, trial=TRIAL, scan=SCAN, seed=SEED):
    """One subject's model and its whole scan: `(name, model, raw_mu, nd_idx, colors)`.

    Mirrors `validation_pareto.plot_validation`'s own `model`/`X`/`mu`/`raw_mu`,
    so the front drawn here and the predictions read off `model` below agree.
    """
    dataset = plr.ExperimentDataset.load(HUMAN_DATA / run / f"{run}.json")
    model = dataset.get_model(trial)

    X = plr.sample_actions(model.action_bounds, scan, "sobol", seed)
    mu = model.posterior_at(X)[0]
    raw_mu = model.posterior_at(X, raw=True)[0]
    nd_idx = front_order(mu, raw_mu)
    colors = np.clip(model.objectives.xtransform(X), 0.0, 1.0)

    return SUBJECTS[run], model, raw_mu, nd_idx, colors


def validation_points(run, model, tol=MATCH_TOL):
    """A subject's validation actions, by source: `{source: (predicted, measured)}`.

    `predicted` and `measured` are `(k, 2)`, in the objectives' own units, so
    they plot directly against the front panel `small_front_axes` builds for the
    same `model`.
    """
    validation = plr.ExperimentDataset.load(HUMAN_DATA / run / "evaluation.json")
    names = model.objectives.names

    actions = validation.get_actions()
    sources = np.asarray(validation.get_sources())
    predicted, _ = model.posterior_at(actions, raw=True)
    measured = measured_at(validation.get_objectives(), actions, names, tol)

    return {source: (predicted[sources == source], measured[sources == source])
            for source in SOURCE_COLORS}


def marker(kind, point, color, size):
    """One `Star` (predicted) or `Square` (measured) at `point`, matching
    `validation_pareto.py`'s own star/square marker convention."""
    shape = (Star(n=5, outer_radius=size, color=color, fill_color=color,
                  fill_opacity=1.0, stroke_color=INK, stroke_width=MARKER_STROKE)
             if kind == "predicted" else
             Square(side_length=size, color=color, fill_color=color,
                    fill_opacity=1.0, stroke_color=INK, stroke_width=MARKER_STROKE))
    return shape.move_to(point)


def legend(colors):
    """One swatch per validation source, `SOURCE_LABELS` beside it, in a row."""
    rows = VGroup()
    for source, color in colors.items():
        swatch = Line(np.zeros(3), np.array([0.5, 0, 0]), color=color, stroke_width=5)
        text = Tex(SOURCE_LABELS[source], font_size=LEGEND_SIZE, color=INK)
        text.next_to(swatch, buff=0.15)
        rows.add(VGroup(swatch, text))
    rows.arrange(buff=LEGEND_GAP)
    return rows.move_to([0.0, LEGEND_Y, 0])


class SubjectValidationScene(Scene):
    """Every subject's front, then its Pareto and anti-Pareto validation points:
    every subject's stars together, then every subject's squares and joining
    lines together, one source at a time."""

    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        self.play(Write(title, run_time=1.0))

        panels = VGroup()
        squares = {source: VGroup() for source in SOURCE_COLORS}
        stars = {source: VGroup() for source in SOURCE_COLORS}
        segments = {source: VGroup() for source in SOURCE_COLORS}
        colors = {source: ManimColor(hex_) for source, hex_ in SOURCE_COLORS.items()}

        for run, x in zip(RUNS, COL_X):
            name, model, raw_mu, nd_idx, cloud_colors = subject_panel_data(run)
            points = validation_points(run, model)

            f_axes, f_labels = small_front_axes(
                (x, FRONT_Y),
                np.vstack([raw_mu, *(v for pair in points.values() for v in pair)]),
                width=FRONT_W, height=FRONT_H, pad=PAD,
            )
            cloud, line, markers = front_cloud(f_axes, raw_mu, nd_idx, cloud_colors)
            name_label = Tex(name, font_size=NAME_LABEL_SIZE, color=INK)
            name_label.next_to(f_axes, UP, buff=0.2)
            panels.add(VGroup(f_axes, f_labels, cloud, line, markers, name_label))

            for source, (predicted, measured) in points.items():
                for p, m in zip(predicted, measured):
                    p2, m2 = f_axes.c2p(*p), f_axes.c2p(*m)
                    squares[source].add(marker("measured", m2, colors[source], SQUARE_SIZE))
                    stars[source].add(marker("predicted", p2, colors[source], STAR_SIZE))
                    segments[source].add(Line(p2, m2, color=colors[source],
                                              stroke_width=LINE_STROKE).set_z_index(-1))

        self.play(Create(panels))
        key = legend(colors)
        self.play(FadeIn(key))
        self.wait(HOLD)

        for source in SOURCE_COLORS:
            self.play(FadeIn(stars[source]))
            self.wait(HOLD)
            self.play(FadeIn(squares[source]), Create(segments[source]))
            self.wait(HOLD)

        self.play(FadeOut(VGroup(title, key, panels, *squares.values(),
                                 *stars.values(), *segments.values())))
