"""One run's inferred Pareto front, animated to video: every fitted trial as a
frame, in objective space beside the actions that produced it, closing on a
point gliding end to end along the final trial's Pareto set and front.

Reuses `plot_fit_gif.py`'s posterior-reading and frame-drawing logic, but
writes an mp4 through ffmpeg instead of a gif through Pillow, at the same
1920x1080 (16:9) resolution and aspect ratio as the manim scenes under
`video/media/videos`, under a large Computer Modern title set in the manim
house style -- `video/style.py`'s own ink and background -- even though the
plot itself stays matplotlib.

After the last trial's hold, one bolded point glides along the final trial's
non-dominated action-space path (the Pareto *set*, on `a_ax`) while a second,
paired point glides along the corresponding objective-space path (the Pareto
*front*, on `p_ax`) -- the same fractional progress on both, so the two always
name the same action. The glide is smoothly eased (`smoothstep`) and
interpolated by arc length in action space, which is why `FPS` is much higher
here than a once-per-trial cadence would otherwise need; `TRIAL_SECONDS` holds
each trial frame on screen for as long as the old `fps=1.5` cadence did.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import FuncFormatter

import pypolar as plr
from hilo.analysis.plot_fit_gif import (
    SCAN,
    SEED,
    draw_frame,
    front_order,
    measured_pairs,
    padded_limits,
)
from video.style import BG, COMFORT_COLOR, COST_COLOR, INK, STAR_COLOR

DATASET  = Path('human_data/MB05/MB05.json')
OUTPUT   = Path('hilo/output/subject3_optimization.mp4')
TITLE    = 'Subject 3 Optimization'
TRIAL_LO = 0
TRIAL_HI = 23   # inclusive, 0-indexed; displayed as Trial 1 through Trial 24

FPS           = 24.0
TRIAL_SECONDS = 2 / 3   # on-screen time per trial frame, matching the old fps=1.5 cadence
DPI       = 150
FIGSIZE   = (12.8, 7.2)   # 1920x1080 at DPI, the manim scenes' own 16:9 frame
FONT      = 'cmr10'       # Computer Modern, matching pypolar's own house style
MATH_FONT = 'cm'
TITLE_SIZE = 40

# The front panel a little wider than the pareto-set panel, both sized against
# tight margins rather than constrained_layout's, which left both panels small.
# right is well short of 1.0 -- mplot3d draws the z-axis label along the right
# edge of the 3D box, and a_ax is itself the figure's rightmost panel, so a
# tighter margin here clips it.
WIDTH_RATIOS = (1.15, 1.0)
MARGINS = dict(left=0.05, right=0.92, top=0.85, bottom=0.11, wspace=0.1)

DELAY_SECONDS = 0.005   # hip_delay_idx (the action panel's z-axis) is a 200 Hz tick count, not seconds

# A little larger than dress_axis's own house defaults (12/14/16), to read at
# 1080p. AXIS_LINEWIDTH matches manim's own AXIS_CONFIG stroke_width, in points.
TICK_SIZE     = 15
LABEL_SIZE    = 19
AX_TITLE_SIZE = 21
AXIS_LINEWIDTH = 2.5

HOLD_SECONDS = 5.0   # how long the last trial's frame stays on screen before the glide

GLIDE_SECONDS      = 4.0   # time for the point to cross the front end to end
GLIDE_HOLD_SECONDS = 1.0   # pause at each end of the glide
GLIDE_POINT_SIZE   = 260   # well above the front's own marker sizes, to read as "the" point
GLIDE_POINT_COLOR  = STAR_COLOR.to_hex()   # the manim scenes' own optimum accent color


def smoothstep(t):
    """Ease `t` in `[0, 1]` with zero velocity at both ends (`3t^2 - 2t^3`)."""
    return 3 * t**2 - 2 * t**3


def glide_path(action_front, objective_front):
    """A function of fractional progress `s` in `[0, 1]` -> `(action, objective)`,
    the point at that progress along `action_front`'s polyline, interpolated by
    arc length in action space, with `objective_front` interpolated by the same
    per-segment fraction so the two paths always name the same point.

    Args:
        action_front: (k, d) raw actions on the front, ordered along it.
        objective_front: (k, m) the same front's raw objective values, in the
            same order.
    """
    seg_lengths = np.linalg.norm(np.diff(action_front, axis=0), axis=1)
    cum_length  = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total       = cum_length[-1]

    def at(s):
        target = np.clip(s, 0.0, 1.0) * total
        i = min(max(np.searchsorted(cum_length, target, side='right') - 1, 0),
               len(seg_lengths) - 1)
        t = 0.0 if seg_lengths[i] == 0 else (target - cum_length[i]) / seg_lengths[i]
        action    = action_front[i]    + t * (action_front[i + 1]    - action_front[i])
        objective = objective_front[i] + t * (objective_front[i + 1] - objective_front[i])
        return action, objective

    return at


def _format_delay_axis(ax):
    """Relabel `ax`'s z-axis (the action panel's 'Delay' dimension) in seconds
    and rescale its tick labels by `DELAY_SECONDS`, without touching the
    plotted points themselves -- `hip_delay_idx` is a raw 200 Hz tick count, so
    only the displayed numbers need converting, not the action data or the
    GP/action_bounds it's drawn against. Run after `dress_axis`, which would
    otherwise reset the tick formatter."""
    ax.set_zlabel('Delay (s)')
    ax.zaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val * DELAY_SECONDS:g}'))


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
        fps: frame granularity; each trial's on-screen time is fixed by
            `TRIAL_SECONDS`, the closing hold by `HOLD_SECONDS`, and the
            closing glide by `GLIDE_SECONDS` / `GLIDE_HOLD_SECONDS` -- a higher
            `fps` only makes the glide itself smoother.
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
    trial_frames = max(1, round(TRIAL_SECONDS * fps))
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
            _format_delay_axis(a_ax)
            for _ in range(trial_frames):
                writer.grab_frame()

        # hold the last trial's frame, already drawn above, before the glide
        for _ in range(round(HOLD_SECONDS * fps)):
            writer.grab_frame()

        nd_idx = front_order(mu, raw_mu)
        if len(nd_idx) >= 2:
            at = glide_path(X[nd_idx], raw_mu[nd_idx])
            glide_frames      = max(1, round(GLIDE_SECONDS * fps))
            glide_hold_frames = max(1, round(GLIDE_HOLD_SECONDS * fps))
            p_point = a_point = None

            def draw_point(s):
                nonlocal p_point, a_point
                action, objective = at(s)
                if p_point is not None:
                    p_point.remove()
                    a_point.remove()
                p_point = p_ax.scatter(*objective, s=GLIDE_POINT_SIZE, c=GLIDE_POINT_COLOR,
                                       edgecolors='black', linewidths=2.0, zorder=5)
                a_point = a_ax.scatter(*action, s=GLIDE_POINT_SIZE, c=GLIDE_POINT_COLOR,
                                       edgecolors='black', linewidths=2.0, zorder=5,
                                       depthshade=False)

            draw_point(0.0)
            for _ in range(glide_hold_frames):
                writer.grab_frame()
            for frame in range(1, glide_frames + 1):
                draw_point(smoothstep(frame / glide_frames))
                writer.grab_frame()
            for _ in range(glide_hold_frames):
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
