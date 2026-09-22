"""The same three Pareto sets, swept from their comfortable end to their efficient one.

A point on a front is one controller, and the front is the ordered list of them:
`front_order` sorts by metabolic cost, so a set's first action is its most
efficient and its last is its most comfortable. Sweeping that ordering lights the
same trade-off in all three subjects at once, which is what says whether they
agree on where a comfortable controller sits.
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
    Scene,
    Tex,
    Triangle,
    VGroup,
    ValueTracker,
    Write,
    linear,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.front import action_axes, action_labels, orient  # noqa: E402
from video.pareto_sets import (  # noqa: E402
    BOX_LEN,
    BOX_X,
    BOX_Y,
    LABEL_HALO,
    MARKER_RADIUS,
    SUBJECT_COLORS,
    legend,
    load_sets,
    subject_set,
)
from video.style import (  # noqa: E402
    ARROW_COLOR,
    AXIS_LABEL_SIZE,
    BG,
    INK,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
)

TITLE = "Trading Comfort for Efficiency"

# Where the sweep is along a front, as a fraction of it: 1 is the comfortable
# end and 0 the efficient one, which is the order `front_order` leaves the rows
# in. Stating it as a fraction rather than a count is what makes one sweep mean
# the same thing on fronts of 26, 51 and 29 points.
START, END = 1.0, 0.0
WINDOW = 0.07

LIT_RADIUS = MARKER_RADIUS * 1.9
DIM_OPACITY = 0.22

# The track the sweep is read off, under the box.
TRACK_Y = -3.2
TRACK_HALF = 2.4
TRACK_STROKE = 2.5
TRACK_LABEL_BUFF = 0.22
HANDLE_SIZE = 0.16

SWEEP_TIME = 7.0
HOLD = 1.0


def positions(actions):
    """`(k,)` each action's place along its own front, from 0 to 1.

    A one-point front has no spread to divide by and sits at the efficient end.
    """
    if len(actions) < 2:
        return np.zeros(len(actions))

    return np.linspace(0.0, 1.0, len(actions))


def sweep_updater(position, tracker):
    """An updater lighting one point while the sweep is within `WINDOW` of it.

    A lit point is drawn at `LIT_RADIUS` and full opacity, a dim one at
    `MARKER_RADIUS` and `DIM_OPACITY`. Both are set absolutely rather than
    stepped, so a point that is lit for several frames does not grow each one.
    """
    def update(dot):
        lit = abs(position - tracker.get_value()) <= WINDOW
        dot.width = 2 * (LIT_RADIUS if lit else MARKER_RADIUS)
        dot.set_opacity(1.0 if lit else DIM_OPACITY)

    return update


def track(tracker):
    """The sweep's own readout: a line between the two ends, and a sliding handle.

    The handle is placed by an updater rather than animated separately, so it
    cannot drift out of step with the points it explains.
    """
    line = Line([BOX_X - TRACK_HALF, TRACK_Y, 0], [BOX_X + TRACK_HALF, TRACK_Y, 0],
                color=ARROW_COLOR, stroke_width=TRACK_STROKE)
    left = Tex("more comfortable", font_size=AXIS_LABEL_SIZE, color=INK)
    right = Tex("more efficient", font_size=AXIS_LABEL_SIZE, color=INK)
    left.next_to(line.get_start(), DOWN, buff=TRACK_LABEL_BUFF)
    right.next_to(line.get_end(), DOWN, buff=TRACK_LABEL_BUFF)

    handle = Triangle(color=INK, fill_opacity=1.0, stroke_width=0)
    handle.height = HANDLE_SIZE
    # The sweep runs from `START` to `END`, and the track is read left to right,
    # so the handle's place along it is the distance already travelled.
    handle.add_updater(lambda m: m.move_to(
        line.point_from_proportion(
            abs(START - tracker.get_value()) / abs(START - END))
        + np.array([0.0, HANDLE_SIZE, 0.0])))

    return VGroup(line, left, right), handle


class TradeoffScene(Scene):
    """The three Pareto sets, with a window sweeping from comfort to efficiency."""

    def construct(self):
        names, sets, box = load_sets()

        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        axes = action_axes(box, length=BOX_LEN)
        frame = axes.copy()
        orient(axes, center=(BOX_X, BOX_Y))
        labels = action_labels(axes)
        labels.set_stroke(BG, width=LABEL_HALO, background=True)
        labels.set_z_index(1)

        key = legend(names)
        tracker = ValueTracker(START)

        # The updaters are attached before anything is drawn, so the sets fade in
        # already showing the comfortable end rather than snapping to it.
        drawn = VGroup()
        for actions, color in zip(sets, SUBJECT_COLORS):
            markers = subject_set(frame, actions, color)
            for dot, position in zip(markers, positions(actions)):
                dot.add_updater(sweep_updater(position, tracker))
            drawn.add(markers)

        rail, handle = track(tracker)

        self.play(Write(title, run_time=1.0))
        self.play(Create(axes), FadeIn(labels), FadeIn(key))
        self.play(FadeIn(drawn), FadeIn(rail), FadeIn(handle))
        self.wait(HOLD)

        self.play(tracker.animate.set_value(END), run_time=SWEEP_TIME, rate_func=linear)
        self.wait(HOLD)

        handle.clear_updaters()
        for markers in drawn:
            for dot in markers:
                dot.clear_updaters()
        self.play(FadeOut(VGroup(title, axes, labels, key, drawn, rail, handle)))
