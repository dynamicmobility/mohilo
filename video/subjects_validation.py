"""Three subjects' fronts, then their held-out validation points and the
hypervolume those points attain, then a close look at what "ordering" a front
even means.

Draws the same panel `subjects_grid.py`'s top row does -- one subject's
inferred front per column -- and then replays `scripts/icra/validation_pareto.py`
on top of it, one source at a time: every subject's stars (the GP's
prediction) first, then every subject's squares (what was measured) together
with the line joining each to its own star, and only then the same two beats
for the anti-Pareto actions. Only once every star and square is on screen does
the hypervolume appear -- the Pareto region first, then morphed into the
anti-Pareto one in place, so the shrink from one to the other is the thing a
viewer watches happen rather than two static shapes they have to compare.

Last, once that whole comparison has faded out, `_ordering_section` zooms into
Subject 1's Pareto panel alone -- posterior cloud, stars and squares, joined
by the same star-to-square lines as before -- drawn twice side by side, then
numbered on top of that: by cost on the left and by comfort on the right, so
a viewer sees that "point 1" is a different square depending on which
objective is doing the ordering.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Create,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    Polygon,
    Scene,
    Square,
    Star,
    Tex,
    Transform,
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
from video.style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    BG,
    COMFORT_COLOR,
    COST_COLOR,
    INK,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
)
from video.subjects_grid import (  # noqa: E402
    HUMAN_DATA,
    NAME_LABEL_SIZE,
    RUNS,
    front_cloud,
    small_front_axes,
)

TITLE = "Validation Against Held-Out Actions"

TRIAL = -1        # the optimization trial whose GP is drawn, and predicts
SCAN = 2**12       # Sobol points the front is read off -- matches
                   # `scripts/icra/human_pareto.py`'s own `SCAN`
SEED = 95

COL_X = (-4.7, 0.0, 4.7)
FRONT_Y = -0.4
FRONT_W, FRONT_H = 3.9, 3.6
PAD = 0.08

STAR_SIZE = 0.17
SQUARE_SIZE = 0.13
MARKER_STROKE = 1.5
LINE_STROKE = 3.0

HV_MARGIN = 0.1     # reference point's gap past the worst square, as a
                    # fraction of that source's own range -- `reference_point`'s
                    # own default
HV_OPACITY = 0.18

LEGEND_Y = 2.55
LEGEND_ROW_GAP = 0.28
LEGEND_GAP = 0.55
LEGEND_SIZE = 22
HV_SWATCH_SIZE = 0.22

HOLD = 1.2

ZOOM_TITLE = "Subject 1 --- Ordering the Same Front Two Ways"
ZOOM_LEFT_X, ZOOM_RIGHT_X = -3.3, 3.3
ZOOM_Y = -0.2
ZOOM_W, ZOOM_H = 5.4, 4.8
ZOOM_PAD = 0.18
ZOOM_PANEL_TITLE_SIZE = 28
RANK_SIZE = 30
RANK_OFFSET = np.array([0.08, 0.32, 0.0])   # up and slightly right of each square
RANK_HALO = 6.0


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


def hypervolume_reference(points, maximize, margin=HV_MARGIN):
    """One subject's shared hypervolume reference: `(m,)`, in raw units.

    Mirrors `hilo/analysis/validation.py`'s own `ref_point` — `plr.
    reference_point_from_objectives(pareto_pts + antipareto_pts, margin=0.1)` —
    which reads its range from *both* sources' measurements added together, so
    one subject's Pareto and anti-Pareto hypervolumes share one reference and
    are directly comparable; only the reference varies subject to subject.
    `points` is `{source: (predicted, measured)}` as `validation_points`
    returns it, and `reference_point` on the stacked `measured` columns is the
    same computation `reference_point_from_objectives` does off each
    objective's own `ydata` range.
    """
    measured = np.vstack([m for _, m in points.values()])
    return plr.reference_point(values=measured, maximize=maximize, margin=margin)


def nondominated_front(measured, maximize):
    """The non-dominated subset of `measured`, `(k, 2)`, sorted ascending
    along the first (cost) column.

    Shared by `hypervolume_shape` and the ordering section below, so both read
    the same front off the same squares. Non-domination is read in
    maximization space, which is what `plr.get_nondominated_tol` wants (larger
    is better in every column), so cost is not simply sorted on: `maximize`
    says which raw columns need flipping first.
    """
    sign = np.where(maximize, 1.0, -1.0)
    nd_idx = plr.get_nondominated_tol(measured * sign)
    return measured[nd_idx[np.argsort(measured[nd_idx, 0])]]


def rank_labels(axes, front, ranks, color, offset=RANK_OFFSET):
    """One number per front point, `ranks[i]` beside `front[i]`, haloed
    against `BG` so a label stays legible over the line or a square it lands
    near."""
    labels = VGroup()
    for p, r in zip(front, ranks):
        text = Tex(str(r), font_size=RANK_SIZE, color=color)
        text.move_to(axes.c2p(*p) + offset)
        text.set_stroke(BG, width=RANK_HALO, background=True)
        labels.add(text)
    return labels


def hypervolume_shape(measured, maximize, ref, axes, color):
    """One source's hypervolume region, shaded on `axes`.

    `measured` is that source's `(k, 2)` raw squares alone -- not the GP's
    scan -- so this reads what was actually attained rather than what the
    model believes. `ref` is shared with the other source on the same subject
    -- see `hypervolume_reference` -- so it is taken as an argument rather
    than read off `measured` alone.

    The returned `Polygon` is the staircase union of the axis-aligned
    rectangles each non-dominated square opens toward `ref`, which is the
    hypervolume itself -- not the smooth region a straight line through the
    squares would enclose -- drawn as a shaded region a viewer can compare
    between sources and subjects by eye.
    """
    front = nondominated_front(measured, maximize)

    verts = [(front[0, 0], ref[1]), tuple(front[0])]
    for x, y in front[1:]:
        verts.append((x, verts[-1][1]))
        verts.append((x, y))
    verts.append((ref[0], verts[-1][1]))
    verts.append((ref[0], ref[1]))

    return Polygon(*(axes.c2p(*v) for v in verts), stroke_width=0,
                   fill_color=color, fill_opacity=HV_OPACITY).set_z_index(-1)


def legend_marker(kind):
    """One neutral `Star`/`Square` icon, white-filled with an ink edge -- the
    same generic-marker convention `validation_pareto.py`'s own
    `legend_handles` draws its proxies in (`mfc='white', mec='black'`), since
    a legend icon means the *shape*, not any one source's color."""
    return (Star(n=5, outer_radius=STAR_SIZE, color=WHITE, fill_color=WHITE,
                fill_opacity=1.0, stroke_color=INK, stroke_width=MARKER_STROKE)
            if kind == "predicted" else
            Square(side_length=SQUARE_SIZE, color=WHITE, fill_color=WHITE,
                  fill_opacity=1.0, stroke_color=INK, stroke_width=MARKER_STROKE))


def legend(colors):
    """Two rows: `SOURCE_LABELS` in their own colors, then what a star, a
    square and a shaded region each mean, generic across sources."""
    source_row = VGroup()
    for source, color in colors.items():
        swatch = Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=5)
        text = Tex(SOURCE_LABELS[source], font_size=LEGEND_SIZE, color=INK)
        text.next_to(swatch, buff=0.15)
        source_row.add(VGroup(swatch, text))
    source_row.arrange(buff=LEGEND_GAP)

    marker_row = VGroup()
    for icon, label in (
        (legend_marker("predicted"), "GP prediction"),
        (legend_marker("measured"), "measured"),
        (Square(side_length=HV_SWATCH_SIZE, fill_color=INK, fill_opacity=HV_OPACITY,
                stroke_width=0), "hypervolume"),
    ):
        text = Tex(label, font_size=LEGEND_SIZE, color=INK)
        text.next_to(icon, buff=0.15)
        marker_row.add(VGroup(icon, text))
    marker_row.arrange(buff=LEGEND_GAP)

    key = VGroup(source_row, marker_row).arrange(DOWN, buff=LEGEND_ROW_GAP)
    return key.move_to([0.0, LEGEND_Y, 0])


class SubjectValidationScene(Scene):
    """Every subject's front, then its Pareto and anti-Pareto validation points
    -- stars together, then squares and joining lines together, one source at
    a time -- and only once every point is down, the hypervolume: Pareto's
    shown, then morphed into anti-Pareto's, so the shrink reads as motion."""

    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        self.play(Write(title, run_time=1.0))

        panels = VGroup()
        squares = {source: VGroup() for source in SOURCE_COLORS}
        stars = {source: VGroup() for source in SOURCE_COLORS}
        segments = {source: VGroup() for source in SOURCE_COLORS}
        hv_areas = {source: VGroup() for source in SOURCE_COLORS}
        colors = {source: ManimColor(hex_) for source, hex_ in SOURCE_COLORS.items()}

        subject1 = None    # `RUNS[0]`'s panel data, kept for the ordering
                           # section below, which re-draws the same posterior
                           # cloud and Pareto points rather than reusing any
                           # of the `panels` mobjects (those get faded out)

        for run, x in zip(RUNS, COL_X):
            name, model, raw_mu, nd_idx, cloud_colors = subject_panel_data(run)
            points = validation_points(run, model)
            if run == RUNS[0]:
                subject1 = (model, raw_mu, nd_idx, cloud_colors, points["pareto"])

            f_axes, f_labels = small_front_axes(
                (x, FRONT_Y),
                np.vstack([raw_mu, *(v for pair in points.values() for v in pair)]),
                width=FRONT_W, height=FRONT_H, pad=PAD,
            )
            cloud, line, markers = front_cloud(f_axes, raw_mu, nd_idx, cloud_colors)
            name_label = Tex(name, font_size=NAME_LABEL_SIZE, color=INK)
            name_label.next_to(f_axes, UP, buff=0.2)
            panels.add(VGroup(f_axes, f_labels, cloud, line, markers, name_label))

            ref = hypervolume_reference(points, model.objectives.maximize)
            for source, (predicted, measured) in points.items():
                for p, m in zip(predicted, measured):
                    p2, m2 = f_axes.c2p(*p), f_axes.c2p(*m)
                    squares[source].add(marker("measured", m2, colors[source], SQUARE_SIZE))
                    stars[source].add(marker("predicted", p2, colors[source], STAR_SIZE))
                    segments[source].add(Line(p2, m2, color=colors[source],
                                              stroke_width=LINE_STROKE).set_z_index(-1))

                hv_areas[source].add(hypervolume_shape(
                    measured, model.objectives.maximize, ref, f_axes, colors[source]))

        self.play(Create(panels))
        key = legend(colors)
        self.play(FadeIn(key))
        self.wait(HOLD)

        for source in SOURCE_COLORS:
            self.play(FadeIn(stars[source]))
            self.wait(HOLD)
            self.play(FadeIn(squares[source]), Create(segments[source]))
            self.wait(HOLD)

        # `SOURCE_COLORS` is insertion-ordered pareto, antipareto, so `first`
        # is drawn and `second` is what it morphs into -- the shrink itself is
        # the point, not either shape alone.
        first, second = SOURCE_COLORS
        self.play(FadeIn(hv_areas[first]))
        self.wait(HOLD)
        self.play(Transform(hv_areas[first], hv_areas[second]))
        self.wait(HOLD)

        self.play(FadeOut(VGroup(title, key, panels, *squares.values(),
                                 *stars.values(), *segments.values(),
                                 hv_areas[first])))

        self._ordering_section(*subject1, colors["pareto"])

    def _ordering_section(self, model, raw_mu, nd_idx, cloud_colors, pareto, color):
        """Subject 1's Pareto panel -- posterior cloud, stars, squares and the
        lines joining each star to its own square -- duplicated left and
        right, then numbered by cost on the left and by comfort on the right,
        so the same squares read as two different rankings depending on which
        axis you sort by.
        """
        predicted, measured = pareto
        front = nondominated_front(measured, model.objectives.maximize)

        # Front is already sorted ascending cost, so "1" (best, since cost is
        # minimized) is just its row order. Comfort is maximized, so its best
        # is the largest value; `comfort_ranks[i]` is where row `i` falls in
        # that descending order, independent of the cost ordering.
        cost_ranks = np.arange(1, len(front) + 1)
        comfort_ranks = np.empty(len(front), dtype=int)
        comfort_ranks[np.argsort(-front[:, 1])] = np.arange(1, len(front) + 1)

        section_title = Tex(ZOOM_TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        section_title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        self.play(Write(section_title, run_time=1.0))

        combined = np.vstack([raw_mu, predicted, measured])
        left_axes, left_labels = small_front_axes(
            (ZOOM_LEFT_X, ZOOM_Y), combined, width=ZOOM_W, height=ZOOM_H, pad=ZOOM_PAD)
        right_axes, right_labels = small_front_axes(
            (ZOOM_RIGHT_X, ZOOM_Y), combined, width=ZOOM_W, height=ZOOM_H, pad=ZOOM_PAD)
        left_title = Tex("Ordered by Metabolic Cost", font_size=ZOOM_PANEL_TITLE_SIZE, color=INK)
        left_title.next_to(left_axes, UP, buff=0.25)
        right_title = Tex("Ordered by Comfort", font_size=ZOOM_PANEL_TITLE_SIZE, color=INK)
        right_title.next_to(right_axes, UP, buff=0.25)

        self.play(Create(VGroup(left_axes, left_labels, right_axes, right_labels)),
                  FadeIn(left_title, right_title))

        def panel_points(axes):
            """That axes' posterior cloud, stars, squares and star-to-square
            lines -- the same four groups the main section draws, rebuilt
            here since `panels`/`stars`/`squares`/`segments` there were
            already faded out."""
            cloud, cloud_line, cloud_markers = front_cloud(axes, raw_mu, nd_idx, cloud_colors)
            squares, stars, segments = VGroup(), VGroup(), VGroup()
            for p, m in zip(predicted, measured):
                p2, m2 = axes.c2p(*p), axes.c2p(*m)
                squares.add(marker("measured", m2, color, SQUARE_SIZE))
                stars.add(marker("predicted", p2, color, STAR_SIZE))
                segments.add(Line(p2, m2, color=color, stroke_width=LINE_STROKE).set_z_index(-1))
            return VGroup(cloud, cloud_line, cloud_markers), squares, stars, segments

        left_cloud, left_squares, left_stars, left_segments = panel_points(left_axes)
        right_cloud, right_squares, right_stars, right_segments = panel_points(right_axes)

        self.play(FadeIn(left_cloud), FadeIn(right_cloud))
        self.wait(HOLD)
        self.play(FadeIn(left_stars), FadeIn(right_stars))
        self.wait(HOLD)
        self.play(FadeIn(left_squares), Create(left_segments),
                  FadeIn(right_squares), Create(right_segments))
        self.wait(HOLD)

        # Each panel's numbers take the color of the objective doing the
        # ordering -- `COST_COLOR`/`COMFORT_COLOR` are the same ones
        # `small_front_axes` already colored that panel's own axis with --
        # rather than the Pareto source color the markers use.
        cost_labels = rank_labels(left_axes, front, cost_ranks, COST_COLOR)
        comfort_labels = rank_labels(right_axes, front, comfort_ranks, COMFORT_COLOR)

        self.play(FadeIn(cost_labels))
        self.wait(HOLD)
        self.play(FadeIn(comfort_labels))
        self.wait(2 * HOLD)

        self.play(FadeOut(VGroup(
            section_title, left_axes, left_labels, left_title, left_cloud, left_squares,
            left_stars, left_segments, cost_labels, right_axes, right_labels, right_title,
            right_cloud, right_squares, right_stars, right_segments, comfort_labels)))
