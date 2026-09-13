"""One subject's Pareto set as a posterior cloud in action space.

Every joint posterior draw of the final GP has its own Pareto set over a Sobol
scan of the action box, and the fraction of draws whose set holds an action is
that action's probability of Pareto optimality, p(x). The Vorob'ev mean set
{x : p(x) >= beta} is drawn opaque over the cloud of every other action, and the
Vorob'ev deviation, the expected disagreement between one draw's set and it, is
reported beside it.

The run is `<data-dir>/<subject>/<subject>.json`. `--layout` draws one 3D axes,
the pairwise 2D projections, or both, as in `subject_actions.py`.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import pypolar as plr
from pypolar.utils.plotting import FONT
from scripts.icra.human_pareto import content_bbox
from scripts.icra.subject_actions import AZIM, ELEV, PANELS, dress_panel

SUBJECT     = 'MB02'
DATA_DIR    = Path('human_data')
OUTPUT_DIR  = Path('hilo/output/')

TRIAL       = -1        # the optimization trial whose GP is drawn
SCAN        = 2**12     # Sobol points each draw's Pareto set is read off; a draw holds an (n, n) covariance
DRAWS       = 512       # joint posterior draws; p(x) has standard error at most 0.5 / sqrt(DRAWS)
CUTOFF      = 0.02      # smallest p(x) drawn in the cloud
SEED        = 95        # seeds both the Sobol scan and torch's draws
DPI         = 300
CMAP        = 'viridis'
ALPHA       = (0.1, 0.8)    # cloud opacity at p = 0 and p = 1


def pareto_probability(
    model   : plr.DecoupledMOGP,
    actions : np.ndarray,
    draws   : int = DRAWS
):
    """(n,) fraction of `draws` joint posterior draws over the (n, d) raw
    `actions` whose Pareto set holds each action."""
    counts = np.zeros(len(actions))
    for path in model.sample_paths(actions, num_paths=draws):   # (n, m), maximization space
        counts[plr.get_nondominated(path)] += 1

    return counts / draws


def vorobev(p: np.ndarray):
    """Vorob'ev mean and deviation of a random subset of a uniform scan, from
    its (n,) coverage `p`.

    Every scan point stands for the same volume, so a set's size is its count
    and the expected size is sum(p).

    Returns:
        beta: the largest threshold whose level set {p >= beta} holds at least
            sum(p) points.
        mean_set: (n,) bool, the level set at `beta`.
        deviation: expected count of the symmetric difference between one
            draw's set and `mean_set`: p summed outside it plus 1 - p inside.
    """
    expected  = p.sum()
    beta      = np.sort(p)[::-1][max(int(np.ceil(expected)), 1) - 1]
    mean_set  = p >= beta
    deviation = p[~mean_set].sum() + (1 - p[mean_set]).sum()

    return beta, mean_set, deviation


def plot_cloud(
    ax          : plt.Axes,
    actions     : np.ndarray,
    p           : np.ndarray,
    mean_set    : np.ndarray,
    cutoff      : float = CUTOFF
):
    """The (n, 2|3) scan `actions` with p above `cutoff`, colored by p and more
    opaque as it rises, under the Vorob'ev `mean_set` drawn opaque with black
    edges. Returns the `ax`."""
    cmap   = matplotlib.colormaps[CMAP]
    cloud  = (p > cutoff) & ~mean_set
    colors = cmap(p[cloud])
    colors[:, 3] = np.interp(p[cloud], (0, 1), ALPHA)

    shade = {}
    if ax.name == '3d':
        ax.computed_zorder = False   # keeps the mean set above the cloud from every view
        shade = {'depthshade': False}
    ax.scatter(*actions[cloud].T, s=12, c=colors, lw=0, zorder=1, **shade)
    ax.scatter(*actions[mean_set].T, s=30, c=cmap(p[mean_set]), edgecolors='black',
               linewidths=0.6, zorder=2, **shade)

    return ax


def make_figure(
    subject     : str,
    data_dir    : Path  = DATA_DIR,
    path        : Path  = None,
    trial       : int   = TRIAL,
    scan        : int   = SCAN,
    draws       : int   = DRAWS,
    cutoff      : float = CUTOFF,
    seed        : int   = SEED,
    dpi         : int   = DPI,
    elev        : float = ELEV,
    azim        : float = AZIM,
    layout      : str   = '3d'
):
    """One subject's p(x) cloud and Vorob'ev mean set, on the panels
    `PANELS[layout]` names, written to `path`."""
    if path is None:
        path = OUTPUT_DIR / f'{subject}_pareto_cloud.jpg'

    dataset = plr.ExperimentDataset.load(data_dir / subject / f'{subject}.json')
    model   = dataset.get_model(trial)
    actions = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)

    torch.manual_seed(seed)
    p                         = pareto_probability(model, actions, draws)
    beta, mean_set, deviation = vorobev(p)
    expected                  = p.sum()
    print(f'{subject}: {draws} draws over {scan} actions\n'
          f'  expected Pareto set size  {expected:.1f} actions\n'
          f'  Vorob\'ev threshold beta   {beta:.3f} ({mean_set.sum()} actions in the mean set)\n'
          f'  Vorob\'ev deviation        {deviation:.1f} actions ({deviation / expected:.2f} of the expected size)\n'
          f'  actions with p > {cutoff:g}     {(p > cutoff).sum()}')

    bounds = plr.as_bounds(model.action_bounds)
    panels = PANELS[layout]
    fig    = plt.figure(figsize=(6 * len(panels) + 1, 6))
    grid   = fig.add_gridspec(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.05], wspace=0.6)
    for i, dims in enumerate(panels):
        ax = fig.add_subplot(grid[0, i], projection='3d' if len(dims) == 3 else None)
        plot_cloud(ax, actions[:, list(dims)], p, mean_set, cutoff)
        if i == 0:
            handles = [Line2D([], [], ls='', marker='o', mfc='0.5', mec='black', label="Vorob'ev mean set"),
                       Line2D([], [], ls='', marker='o', mfc='0.5', mec='none', alpha=0.4,
                              label=f'$p > {cutoff:g}$')]
            legend = ax.legend(handles=handles, fontsize=12, framealpha=0.9, loc='lower left',
                               bbox_to_anchor=(0, 1.02), ncols=len(handles),
                               title=f"Subject {subject}, Vorob'ev deviation {deviation / expected:.2f}")
            legend.get_title().set_fontfamily(FONT)   # dress_axis restyles legend entries, not the title
        dress_panel(ax, dims, bounds, elev, azim)

    cbar = fig.colorbar(ScalarMappable(Normalize(0, 1), CMAP), cax=fig.add_subplot(grid[0, -1]))
    cbar.ax.axhline(beta, color='black', lw=1.5)
    cbar.set_label('P(Pareto optimal)')
    plr.dress_axis(cbar.ax, label_size=16)
    cbar.ax.grid(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=content_bbox(fig))

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'subject',
        nargs   = '?',
        default = SUBJECT,
        help    = 'subject name, read from <data-dir>/<subject>/<subject>.json'
    )
    p.add_argument(
        '--data-dir',
        type    = Path,
        default = DATA_DIR,
        help    = 'the folder holding one folder per subject'
    )
    p.add_argument(
        '--output',
        type    = Path,
        default = None,
        help    = 'the path to write the figure to'
    )
    p.add_argument(
        '--trial',
        type    = int,
        default = TRIAL,
        help    = 'the optimization trial whose GP is drawn'
    )
    p.add_argument(
        '--scan',
        type    = int,
        default = SCAN,
        help    = 'number of Sobol points each draw is evaluated on; memory grows as its square'
    )
    p.add_argument(
        '--draws',
        type    = int,
        default = DRAWS,
        help    = 'number of joint posterior draws p(x) is estimated from'
    )
    p.add_argument(
        '--cutoff',
        type    = float,
        default = CUTOFF,
        help    = 'smallest probability of Pareto optimality drawn in the cloud'
    )
    p.add_argument(
        '--seed',
        type    = int,
        default = SEED,
        help    = 'the random seed of the Sobol points and the posterior draws'
    )
    p.add_argument(
        '--dpi',
        type    = int,
        default = DPI,
        help    = 'DPI of the generated figure'
    )
    p.add_argument(
        '--elev',
        type    = float,
        default = ELEV,
        help    = '3D view elevation, in degrees'
    )
    p.add_argument(
        '--azim',
        type    = float,
        default = AZIM,
        help    = '3D view azimuth, in degrees'
    )
    p.add_argument(
        '--layout',
        default = '3d',
        choices = list(PANELS),
        help    = 'one 3D axes (3d), the three pairwise 2D projections (2d), or both side by side'
    )

    return p.parse_args()


def main():
    args = parse_args()
    path = make_figure(
        subject     = args.subject,
        data_dir    = args.data_dir,
        path        = args.output,
        trial       = args.trial,
        scan        = args.scan,
        draws       = args.draws,
        cutoff      = args.cutoff,
        seed        = args.seed,
        dpi         = args.dpi,
        elev        = args.elev,
        azim        = args.azim,
        layout      = args.layout
    )
    print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
