"""The method figure, with a box walking from the exoskeleton to each objective."""

import shutil
import subprocess
import sys
from pathlib import Path

from manim import (
    UP,
    Create,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Scene,
    RoundedRectangle,
    Tex,
    Transform,
    Write,
    config,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the `video.style` import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.style import (  # noqa: E402
    BOX_RADIUS,
    INK,
    SCENE_TITLE_SIZE,
    STAR_COLOR,
    TITLE_EDGE_BUFF,
)

TITLE = "Human-Subject Study"

FIGURE_SVG = Path(__file__).resolve().parent / "media" / "images" / "method.svg"

# The figure's own user units: its viewBox is 792 x 335.7 px, and every region
# below is an element's bounding box in that frame, y measured down from the top.
FIGURE_W, FIGURE_H = 792.0, 335.728
EXOSKELETON = (0.0, 0.0, 239.1, 334.5)
METABOLIC = (288.8, 104.3, 175.6, 98.1)
COMFORT = (289.0, 2.8, 174.6, 98.1)

FIGURE_WIDTH = 13.0
# Dropped below center to clear the title.
FIGURE_Y = -0.35
RASTER_HEIGHT_PX = 2000

HIGHLIGHT_COLOR = STAR_COLOR
HIGHLIGHT_STROKE = 5.0
HIGHLIGHT_PAD = 0.08

HOLD = 1.2
END_HOLD = 5.0

def inkscape():
    """The Inkscape executable, from the PATH or from the macOS app bundle."""
    found = shutil.which("inkscape")
    if found:
        return found
    bundled = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError("inkscape is needed to rasterize method.svg")


def rasterize(path, height_px):
    """Export `path` to a transparent PNG of that pixel height, cached by height.

    manim's SVGMobject draws paths only, and this figure is ten embedded
    photographs and matplotlib panels, so it is rendered to an image instead.
    """
    path = Path(path)
    out = Path(config.media_dir) / "raster" / f"{path.stem}_{height_px}px.png"
    if out.exists():
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [inkscape(), "--export-type=png", f"--export-height={height_px}",
         "--export-background-opacity=0", f"--export-filename={out}", str(path)],
        check=True,
    )
    return out


def highlight(figure, region):
    """A rounded box around one region of the figure, in scene units.

    `region` is `(x, y, w, h)` in the figure's own px frame; the figure's drawn
    corners carry the mapping, so it holds at any figure size.
    """
    x, y, w, h = region
    scale = figure.width / FIGURE_W
    left, top = figure.get_left()[0], figure.get_top()[1]
    box = RoundedRectangle(
        width=w * scale + 2 * HIGHLIGHT_PAD,
        height=h * scale + 2 * HIGHLIGHT_PAD,
        corner_radius=BOX_RADIUS,
        stroke_color=HIGHLIGHT_COLOR,
        stroke_width=HIGHLIGHT_STROKE,
        fill_opacity=0.0,
    )
    return box.move_to([left + (x + w / 2) * scale, top - (y + h / 2) * scale, 0])


class MethodScene(Scene):
    """The method figure, with one box visiting the exoskeleton and
    then each of the two objectives it is measured on."""

    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)

        figure = ImageMobject(str(rasterize(FIGURE_SVG, RASTER_HEIGHT_PX)))
        # ImageMobject sizes itself against a fixed 1080p reference, so the width
        # is set here instead and holds at any render resolution.
        figure.width = FIGURE_WIDTH
        figure.move_to([0, FIGURE_Y, 0])

        self.play(Write(title, run_time=1.0))
        self.play(FadeIn(figure))
        self.wait(0.5)

        box = highlight(figure, EXOSKELETON)
        self.play(Create(box))
        self.wait(HOLD + 4.0)

        for region in (METABOLIC, COMFORT):
            self.play(Transform(box, highlight(figure, region)))
            self.wait(HOLD)

        self.wait(END_HOLD)
        self.play(FadeOut(Group(title, figure, box)))
