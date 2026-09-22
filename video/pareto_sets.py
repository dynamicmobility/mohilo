"""Three subjects' Pareto sets in one action space, a color each.

A Pareto *set* is the front read in action space: the controllers that attain
it, rather than the cost and comfort they attain. Drawn together, the three say
how far apart the subjects' optimal controllers sit — which is the question a
per-subject plot cannot answer.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    ManimColor,
    Scene,
    Tex,
    VGroup,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.front import (  # noqa: E402
    SUBJECTS,
    action_axes,
    action_box_faces,
    action_labels,
    action_ticks,
    final_front,
    orient,
    project,
)
from video.style import (  # noqa: E402
    AXIS_LABEL_SIZE,
    BG,
    INK,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
)

HUMAN_DATA = Path(__file__).resolve().parents[1] / "human_data"
RUNS = ("MB02", "MB04", "MB05")
TITLE = "Pareto Sets Across Subjects"

# Only the non-dominated points become mobjects here, not the whole scan, so
# this is the gif's own resolution rather than the one `front.py` cuts down to.
SCAN = 2**12
SEED = 95

# One color per subject. They are their own, not the per-objective colors: a
# color means which subject here, where in every other scene it means which
# objective, and reusing them would read as one of those.
SUBJECT_COLORS = (
    ManimColor("#F4A261"),
    ManimColor("#2EC4B6"),
    ManimColor("#E56BF0"),
)

BOX_LEN = 4.4
BOX_X, BOX_Y = 0.4, -0.9

LABEL_HALO = 8.0

MARKER_RADIUS = 0.085
MARKER_STROKE = 1.0

LEGEND_X, LEGEND_Y = -4.9, 1.9
LEGEND_BUFF = 0.28
LEGEND_DOT_RADIUS = 0.09

STEP_TIME = 1.2
HOLD = 1.5


def load_sets(runs=RUNS, scan=SCAN, seed=SEED):
    """`(names, sets, box)` for every run: its subject name, its front's actions,
    and the `(2, 3)` action box spanning all of them.

    The box is the union rather than any one run's, so a subject measured over a
    narrower range would still be drawn in a frame holding every subject.
    """
    fronts = [final_front(HUMAN_DATA / run / f"{run}.json", scan, seed) for run in runs]
    names = [SUBJECTS[name] for name, _, _ in fronts]
    box = np.vstack([
        np.min([bounds[0] for _, _, bounds in fronts], axis=0),
        np.max([bounds[1] for _, _, bounds in fronts], axis=0),
    ])

    return names, [actions for _, actions, _ in fronts], box


def subject_set(axes, actions, color):
    """One subject's Pareto set, as points in the axes' own frame.

    The points are not joined. Neighbours on the front are neighbours in
    objective space, and two of those can sit far apart in action space, so a
    line through them draws a zig-zag that says more about the ordering than
    about the set.

    They are placed by `project` rather than built and then rotated, which is
    what keeps a marker a circle instead of an ellipse seen edge-on.
    """
    points = project([axes.c2p(*action) for action in actions], center=(BOX_X, BOX_Y))

    return VGroup(*(
        Dot(point, radius=MARKER_RADIUS, color=color,
            stroke_color=INK, stroke_width=MARKER_STROKE)
        for point in points
    ))


def legend(names):
    """A colored dot per subject beside its name, as one stacked block."""
    rows = VGroup()
    for name, color in zip(names, SUBJECT_COLORS):
        dot = Dot(radius=LEGEND_DOT_RADIUS, color=color,
                  stroke_color=INK, stroke_width=MARKER_STROKE)
        text = Tex(name, font_size=AXIS_LABEL_SIZE, color=color)
        rows.add(VGroup(dot, text.next_to(dot, RIGHT, buff=0.18)))
    rows.arrange(DOWN, buff=LEGEND_BUFF, aligned_edge=np.array([-1.0, 0.0, 0.0]))

    return rows.move_to([LEGEND_X, LEGEND_Y, 0])


class ParetoSetsScene(Scene):
    """The three subjects' Pareto sets, drawn one after another in one action box."""

    def construct(self):
        names, sets, box = load_sets()

        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        axes = action_axes(box, length=BOX_LEN)
        # `axes` is rotated into the view, and every set is built on an
        # unrotated copy and rotated the same way, so all of them land in the
        # same box.
        frame = axes.copy()
        faces = orient(action_box_faces(frame), center=(BOX_X, BOX_Y))
        orient(axes, center=(BOX_X, BOX_Y))
        labels = action_labels(axes)
        # The sets crowd the y axis's tip, which is where its name belongs. A
        # background stroke in the scene color keeps the name readable over the
        # points it lands on, rather than moving it away from its own axis.
        labels.set_stroke(BG, width=LABEL_HALO, background=True)
        labels.set_z_index(1)
        ticks = action_ticks(frame, center=(BOX_X, BOX_Y))
        ticks.set_stroke(BG, width=LABEL_HALO, background=True)
        ticks.set_z_index(1)

        key = legend(names)

        self.play(Write(title, run_time=1.0))
        self.play(FadeIn(faces), Create(axes), FadeIn(labels, ticks))

        drawn = VGroup()
        for actions, color, row in zip(sets, SUBJECT_COLORS, key):
            markers = subject_set(frame, actions, color)
            drawn.add(markers)
            self.play(FadeIn(row), FadeIn(markers), run_time=STEP_TIME)

        self.wait(HOLD)
        self.play(FadeOut(VGroup(title, axes, faces, labels, ticks, key, drawn)))
