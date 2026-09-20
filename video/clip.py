"""A video file drawn inside a scene, as an image whose frames advance with the clock.

manim renders mobjects, not video streams, so a clip is decoded to PNG frames
once with ffmpeg and swapped into an ImageMobject one frame at a time.
"""

import subprocess
from pathlib import Path

import numpy as np
from manim import ImageMobject, config
from manim.utils.images import change_to_rgba_array
from PIL import Image


def frame_cache():
    """The directory holding decoded frames, read when called so that the media
    directory `style.py` configures is the one used."""
    return Path(config.media_dir) / "frames"


def units_to_pixels(height_units):
    """Pixel height of something `height_units` tall, at the configured resolution."""
    return int(round(height_units / config.frame_height * config.pixel_height))


def extract_frames(path, height_px, fps, crop=None):
    """Decode `path` to PNG frames of that pixel height, cached by decode settings.

    `crop` is an ffmpeg `w:h:x:y` string, which is how the pillarbox bars around a
    portrait clip are removed. Returns the frame paths in order.
    """
    path = Path(path)
    out_dir = frame_cache() / f"{path.stem}_{crop or 'full'}_{height_px}px_{fps}fps"
    frames = sorted(out_dir.glob("*.png"))
    if frames:
        return frames

    out_dir.mkdir(parents=True, exist_ok=True)
    filters = ([f"crop={crop}"] if crop else []) + [f"fps={fps}", f"scale=-2:{height_px}"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", ",".join(filters),
         str(out_dir / "%05d.png")],
        check=True,
    )
    return sorted(out_dir.glob("*.png"))


class VideoClip(ImageMobject):
    """An ImageMobject showing frame `t * fps` of a video once `play()` is called."""

    def __init__(self, path, height_units, fps=30, crop=None, **kwargs):
        self.frames = extract_frames(path, units_to_pixels(height_units), fps, crop)
        self.fps = fps
        self.elapsed = 0.0
        self.index = 0
        super().__init__(str(self.frames[0]), **kwargs)
        # ImageMobject sizes itself against a fixed 1080p reference, so the
        # height is set here instead and holds at any render resolution.
        self.height = height_units

    @property
    def duration(self):
        """Seconds of footage available."""
        return len(self.frames) / self.fps

    def play(self, loop=False):
        """Advance the drawn frame with the scene clock, holding the last one at the end."""
        self.loop = loop
        self.add_updater(VideoClip._advance)
        return self

    def _advance(self, dt):
        self.elapsed += dt
        index = int(self.elapsed * self.fps)
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)
        if index == self.index:
            return
        self.index = index
        self.pixel_array = change_to_rgba_array(
            np.asarray(Image.open(self.frames[index])), self.pixel_array_dtype
        )
