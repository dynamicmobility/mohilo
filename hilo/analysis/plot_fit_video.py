"""One run's inferred Pareto front, animated to video: every fitted trial as a
frame, in objective space beside the actions that produced it.

Reuses `plot_fit_gif.py`'s posterior-reading and frame-drawing logic, but
writes an mp4 through ffmpeg instead of a gif through Pillow, at the same
1920x1080 (16:9) resolution and aspect ratio as the manim scenes under
`video/media/videos`, under a large Computer Modern title set in the manim
house style -- `video/style.py`'s own ink and background -- even though the
plot itself stays matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

import pypolar as plr
from hilo.analysis.plot_fit_gif import SCAN, SEED, draw_frame, measured_pairs, padded_limits
from video.style import BG, COMFORT_COLOR, COST_COLOR, INK

DATASET  = Path('human_data/MB05/MB05.json')
OUTPUT   = Path('hilo/output/subject3_optimization.mp4')
TITLE    = 'Subject 3 Optimization'
TRIAL_LO = 0
TRIAL_HI = 23   # inclusive, 0-indexed; displayed as Trial 1 through Trial 24

FPS       = 1.5
DPI       = 150
FIGSIZE   = (12.8, 7.2)   # 1920x1080 at DPI, the manim scenes' own 16:9 frame
FONT      = 'cmr10'       # Computer Modern, matching pypolar's own house style
MATH_FONT = 'cm'
TITLE_SIZE = 40

# The front panel a little wider than the pareto-set panel, both sized against
# tight margins rather than constrained_layout's, which left both panels small.
WIDTH_RATIOS = (1.15, 1.0)
MARGINS = dict(left=0.05, right=0.98, top=0.85, bottom=0.11, wspace=0.1)

# A little larger than dress_axis's own house defaults (12/14/16), to read at
# 1080p. AXIS_LINEWIDTH matches manim's own AXIS_CONFIG stroke_width, in points.
TICK_SIZE     = 15
LABEL_SIZE    = 19
AX_TITLE_SIZE = 21
AXIS_LINEWIDTH = 2.5

HOLD_SECONDS = 5.0   # how long the last trial's frame stays on screen at the end


def _colorize_objectives(ax):
    """Color the front panel's axes the way the manim scenes' `front_axes` does:
    metabolic cost in `COST_COLOR`, comfort in `COMFORT_COLOR`, at the same
    stroke width the manim scenes draw their axes with. Run last, after every
    `dress_axis` call, since `dress_axis` recolors both spines `MUTED`."""
    cost, comfort = COST_COLOR.to_hex(), COMFORT_COLOR.to_hex()
    ax.xaxis.label.set_color(cost)
    ax.yaxis.label.set_color(comfort)
    ax.spines['bottom'].set_color(cost)
    ax.spines['left'].set_color(comfort)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis='x', colors=cost)
    ax.tick_params(axis='y', colors=comfort)


def make_video(dataset: plr.ExperimentDataset, path=None, title=TITLE,
               trial_lo=TRIAL_LO, trial_hi=TRIAL_HI, fps=FPS, scan=SCAN, seed=SEED):
    """Every fitted trial in `[trial_lo, trial_hi]` as one frame, written to an mp4.

    Args:
        dataset: the `ExperimentDataset` to draw.
        path: the mp4 to write. Defaults to `OUTPUT`.
        title: the figure's own title, drawn once above both panels.
        trial_lo, trial_hi: the inclusive 0-indexed trial range to draw; each
            frame is labeled `Trial {trial - trial_lo + 1}`, so the range
            always displays as 1-indexed regardless of where it starts.
        fps: frames per second.
        scan: Sobol points the front is read off.
        seed: the scan's seed.

    Returns:
        Where the video went.
    """
    frames = [record.trial for record in dataset.trials
              if record.state_dict is not None and trial_lo <= record.trial <= trial_hi]
    if not frames:
        raise ValueError(f'{dataset.name} has no fitted trial in [{trial_lo}, {trial_hi}]')

    models = [dataset.get_model(trial) for trial in frames]
    names  = models[-1].objectives.names
    X = plr.sample_actions(models[-1].action_bounds, scan, 'sobol', seed)

    posteriors = [(mogp.posterior_at(X)[0],            # maximization space
                   mogp.posterior_at(X, raw=True)[0],  # objectives' own units
                   measured_pairs(mogp))
                  for mogp in models]

    # one axis box for the whole video, spanning every point any frame draws
    low, high = padded_limits(np.vstack(
        [raw_mu for _, raw_mu, _ in posteriors]
        + [measured for _, _, measured in posteriors]
    ))
    action_low, action_high = X.min(axis=0), X.max(axis=0)

    fig  = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG.to_hex())
    fig.suptitle(title, fontsize=TITLE_SIZE, fontfamily=FONT,
                math_fontfamily=MATH_FONT, color=INK.to_hex())
    grid = fig.add_gridspec(1, 2, width_ratios=WIDTH_RATIOS, **MARGINS)
    p_ax = fig.add_subplot(grid[0, 0])
    a_ax = fig.add_subplot(grid[0, 1], projection='3d')

    path = Path(path or OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    with writer.saving(fig, path, dpi=DPI):
        for trial, (mu, raw_mu, measured) in zip(frames, posteriors):
            p_ax.clear()
            a_ax.clear()
            draw_frame(p_ax, a_ax, X, mu, raw_mu, measured, names, dataset.get_model(trial=trial))

            p_ax.set_xlim(low[0], high[0])
            p_ax.set_ylim(low[1], high[1])
            a_ax.set_xlim(action_low[0], action_high[0])
            a_ax.set_ylim(action_low[1], action_high[1])
            a_ax.set_zlim(action_low[2], action_high[2])
            p_ax.set_title(f'Trial {trial - trial_lo + 1}')
            # re-applies the house font to the title just set, at a larger size
            plr.dress_axis(p_ax, tick_size=TICK_SIZE, label_size=LABEL_SIZE,
                           title_size=AX_TITLE_SIZE)
            plr.dress_axis(a_ax, tick_size=TICK_SIZE, label_size=LABEL_SIZE)
            _colorize_objectives(p_ax)
            writer.grab_frame()

        # hold the last trial's frame, already drawn above, for HOLD_SECONDS
        for _ in range(round(HOLD_SECONDS * fps) - 1):
            writer.grab_frame()

    plt.close(fig)

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path, nargs='?', default=DATASET,
                   help='the run to draw, as written by ExperimentDataset.save')
    p.add_argument('--output',    type=Path,  default=OUTPUT, help='the mp4 to write')
    p.add_argument('--title',     type=str,   default=TITLE, help="the figure's own title")
    p.add_argument('--trial-lo',  type=int,   default=TRIAL_LO,
                   help='first 0-indexed trial to draw')
    p.add_argument('--trial-hi',  type=int,   default=TRIAL_HI,
                   help='last 0-indexed trial to draw, inclusive')
    p.add_argument('--fps',       type=float, default=FPS, help='frames per second')
    p.add_argument('--scan',      type=int,   default=SCAN,
                   help='Sobol points the front is read off')

    return p.parse_args()


def main():
    args    = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset)
    path    = make_video(dataset, path=args.output, title=args.title,
                         trial_lo=args.trial_lo, trial_hi=args.trial_hi,
                         fps=args.fps, scan=args.scan)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
