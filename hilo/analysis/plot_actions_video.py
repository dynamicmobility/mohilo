"""Every subject's inferred Pareto set, animated to video: one subject's
action-space front revealed at a time, on one 3D panel.

Mirrors `scripts/icra/subject_actions.py`'s single `'3d'` panel -- the same
fronts, read off the same Sobol scan, in the same Okabe-Ito subject colors --
but written as an mp4 through ffmpeg, built up subject by subject rather than
drawn all at once, in the same style as `hilo/analysis/plot_fit_video.py`:
1920x1080, `video/style.py`'s background and ink, and a Computer Modern title
set through matplotlib rather than manim.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

import pypolar as plr
from scripts.icra.human_pareto import ACTION_LABELS
from scripts.icra.subject_actions import SUBJECT_COLORS, pareto_actions, plot_subject
from scripts.icra.subjects_pareto import SUBJECTS
from video.style import BG, INK

DATA_DIR = Path('human_data')
OUTPUT   = Path('hilo/output/subject_actions.mp4')
TITLE    = 'Subject Comparison'

TRIAL = -1        # the optimization trial whose GP each front is read from
SCAN  = 2**14     # Sobol points the front is read off
SEED  = 95
ELEV  = 25        # 3D view elevation, degrees
AZIM  = -60       # 3D view azimuth, degrees

FPS            = 2.0
REVEAL_SECONDS = 1.0    # how long each newly added subject holds before the next
HOLD_SECONDS   = 10.0   # total dwell time of the final, all-subjects frame

DPI       = 150
FIGSIZE   = (12.8, 7.2)   # 1920x1080 at DPI, the manim scenes' own 16:9 frame
FONT      = 'cmr10'       # Computer Modern, matching pypolar's own house style
MATH_FONT = 'cm'
TITLE_SIZE = 40
TICK_SIZE  = 15
LABEL_SIZE = 19


def make_video(subjects=tuple(SUBJECTS), data_dir=DATA_DIR, path=None, title=TITLE,
               trial=TRIAL, scan=SCAN, seed=SEED, elev=ELEV, azim=AZIM, fps=FPS):
    """Every subject's Pareto-optimal actions, one subject revealed per step,
    written to an mp4.

    Args:
        subjects: subject ids, each read from `<data_dir>/<id>/<id>.json`.
        path: the mp4 to write. Defaults to `OUTPUT`.
        title: the figure's own title, drawn once above the panel.
        trial: the optimization trial whose GP each front is read from.
        scan, seed: the Sobol scan each front is read off.
        elev, azim: the 3D view angle.
        fps: frame granularity; each reveal step and the final hold are timed
            in seconds via `REVEAL_SECONDS` / `HOLD_SECONDS`, not by `fps`
            directly.

    Returns:
        Where the video went.
    """
    if len(subjects) > len(SUBJECT_COLORS):
        raise ValueError(f'{len(subjects)} subjects, but only {len(SUBJECT_COLORS)} colors')

    datasets = [plr.ExperimentDataset.load(data_dir / s / f'{s}.json') for s in subjects]
    boxes    = [plr.as_bounds(d.get_model(trial).action_bounds) for d in datasets]
    bounds   = np.array([np.min([b[0] for b in boxes], axis=0),
                         np.max([b[1] for b in boxes], axis=0)])   # (2, d) union of every box
    fronts   = [pareto_actions(d, trial, scan, seed) for d in datasets]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG.to_hex())
    fig.suptitle(title, fontsize=TITLE_SIZE, fontfamily=FONT,
                math_fontfamily=MATH_FONT, color=INK.to_hex())
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.05)

    path = Path(path or OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    reveal_frames = max(1, round(REVEAL_SECONDS * fps))
    hold_frames   = max(1, round(HOLD_SECONDS * fps))
    with writer.saving(fig, path, dpi=DPI):
        for i in range(len(subjects)):
            ax.clear()
            for j in range(i + 1):
                plot_subject(ax, fronts[j], SUBJECT_COLORS[j])

            for axis, dim in zip(('x', 'y', 'z'), (0, 1, 2)):
                getattr(ax, f'set_{axis}label')(ACTION_LABELS[dim])
                getattr(ax, f'set_{axis}lim')(bounds[:, dim])
            ax.view_init(elev=elev, azim=azim)

            handles = [Line2D([], [], color=SUBJECT_COLORS[j], marker='o', mec='black',
                              label=SUBJECTS[datasets[j].subject])
                      for j in range(i + 1)]
            ax.legend(handles=handles, fontsize=LABEL_SIZE, framealpha=0.9, loc='upper left')

            # after the legend, so its text takes the house font too
            plr.dress_axis(ax, tick_size=TICK_SIZE, label_size=LABEL_SIZE,
                           num_xticks=4, num_yticks=4, num_zticks=4)

            repeats = hold_frames if i == len(subjects) - 1 else reveal_frames
            for _ in range(repeats):
                writer.grab_frame()

    plt.close(fig)

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('subjects', nargs='*', default=list(SUBJECTS),
                   help='subject ids, each read from <data-dir>/<id>/<id>.json')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR,
                   help='the folder holding one folder per subject')
    p.add_argument('--output',   type=Path, default=OUTPUT, help='the mp4 to write')
    p.add_argument('--title',    type=str,  default=TITLE, help="the figure's own title")
    p.add_argument('--trial',    type=int,  default=TRIAL,
                   help='the optimization trial whose GP each front is read from')
    p.add_argument('--scan',     type=int,  default=SCAN,
                   help='number of Sobol points each front is evaluated on')
    p.add_argument('--seed',     type=int,  default=SEED,
                   help='the random seed the Sobol points are generated with')
    p.add_argument('--elev',     type=float, default=ELEV, help='3D view elevation, in degrees')
    p.add_argument('--azim',     type=float, default=AZIM, help='3D view azimuth, in degrees')
    p.add_argument('--fps',      type=float, default=FPS, help='frame granularity')

    return p.parse_args()


def main():
    args = parse_args()
    path = make_video(subjects=args.subjects, data_dir=args.data_dir, path=args.output,
                      title=args.title, trial=args.trial, scan=args.scan, seed=args.seed,
                      elev=args.elev, azim=args.azim, fps=args.fps)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
