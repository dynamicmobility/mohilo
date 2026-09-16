"""Every subject's inferred Pareto set, drawn together in action space.

The actions on each final GP's front are joined in front order, one color per
subject. `--layout` draws them on one 3D axes, on the three 2D projections onto
each pair of action dimensions, or both side by side.

Each subject's optimization run is `<data-dir>/<subject>/<subject>.json`. The
front is read off a Sobol scan of the run's action box, so every subject's
set is drawn in the units the actions were commanded in.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import pypolar as plr
from scripts.icra.human_pareto import ACTION_LABELS, content_bbox
from scripts.icra.validation_pareto import front_order

SUBJECTS    = ['MB02', 'MB04', 'MB05']
RENAME      = {'MB02': 'MB01', 'MB04': 'MB02', 'MB05': 'MB03'}
DATA_DIR    = Path('human_data')
OUTPUT      = Path('hilo/output/subject_actions.jpg')

TRIAL       = -1        # the optimization trial whose GP is drawn
SCAN        = 2**14     # Sobol points the front is read off
SEED        = 95
DPI         = 300
ELEV        = 25        # 3D view elevation, degrees
AZIM        = -60       # 3D view azimuth, degrees

PROJECTIONS = [(0, 1), (0, 2), (1, 2)]                  # action dimension pairs the 2D panels draw
PANELS      = {'3d'  : [(0, 1, 2)],
               '2d'  : PROJECTIONS,
               'both': [(0, 1, 2)] + PROJECTIONS}        # action dimensions of each panel, left to right

SUBJECT_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                  '#E69F00', '#56B4E9', '#F0E442']   # Okabe-Ito, one per subject in order


def pareto_actions(
    dataset : plr.ExperimentDataset,
    trial   : int = TRIAL,
    scan    : int = SCAN,
    seed    : int = SEED
):
    """(k, d) raw actions on one run's inferred front, sorted by the first
    objective's raw value so consecutive rows are neighbors on the front."""
    model     = dataset.get_model(trial)
    X         = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)
    mu, _     = model.posterior_at(X)
    raw_mu, _ = model.posterior_at(X, raw=True)

    return X[front_order(mu, raw_mu)]


def plot_subject(
    ax      : plt.Axes,
    actions : np.ndarray,
    color   : str
):
    """One subject's (k, 2) or (k, 3) Pareto actions as a path of points in
    `color`, on a 2D or 3D `ax` to match. Returns the `ax`."""
    shade = {'depthshade': False} if ax.name == '3d' else {}
    # ax.plot(*actions.T, lw=1.5, color=color, alpha=0.8, zorder=1)
    ax.scatter(*actions.T, s=25, c=color, edgecolors='black', linewidths=0.6,
               zorder=2, **shade)

    return ax


def dress_panel(
    ax      : plt.Axes,
    dims    : tuple[int, ...],
    bounds  : np.ndarray,
    elev    : float = ELEV,
    azim    : float = AZIM
):
    """Label and limit each axis of `ax` by the action dimension `dims` puts on
    it, over the (2, d) `bounds`, and apply the house style. Returns the `ax`."""
    for axis, dim in zip(('x', 'y', 'z'), dims):
        getattr(ax, f'set_{axis}label')(ACTION_LABELS[dim])
        getattr(ax, f'set_{axis}lim')(bounds[:, dim])

    ticks = {'num_xticks': 4, 'num_yticks': 4}
    if ax.name == '3d':
        ax.view_init(elev=elev, azim=azim)
        ticks['num_zticks'] = 4
    else:
        ax.set_box_aspect(1)

    # return plr.dress_axis(ax, tick_size=14, label_size=16, **ticks)
    return plr.dress_axis(
        ax, 
        label_size=22,  
        title_size=26,
        **ticks
    )


def make_figure(
    subjects    : list[str],
    data_dir    : Path = DATA_DIR,
    path        : Path = OUTPUT,
    trial       : int  = TRIAL,
    scan        : int  = SCAN,
    seed        : int  = SEED,
    dpi         : int  = DPI,
    elev        : float = ELEV,
    azim        : float = AZIM,
    layout      : str   = '3d'
):
    """Every subject's Pareto actions over the union of their action boxes, on
    the panels `PANELS[layout]` names, written to `path`."""
    if len(subjects) > len(SUBJECT_COLORS):
        raise ValueError(f'{len(subjects)} subjects, but only {len(SUBJECT_COLORS)} colors')

    datasets = [plr.ExperimentDataset.load(data_dir / s / f'{s}.json') for s in subjects]
    boxes    = [plr.as_bounds(d.get_model(trial).action_bounds) for d in datasets]
    bounds   = np.array([np.min([b[0] for b in boxes], axis=0),
                         np.max([b[1] for b in boxes], axis=0)])   # (2, d) union of every box

    fronts = [pareto_actions(d, trial, scan, seed) for d in datasets]
    for dataset, actions in zip(datasets, fronts):
        print(f'{dataset.subject}: {len(actions)} actions on the front')

    panels = PANELS[layout]
    fig    = plt.figure(figsize=(6 * len(panels), 6))
    for i, dims in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection='3d' if len(dims) == 3 else None)
        for actions, color in zip(fronts, SUBJECT_COLORS):
            plot_subject(ax, actions[:, list(dims)], color)
        if i == 0:
            handles = [Line2D([], [], color=color, marker='o', mec='black', label=f'Subject {RENAME[d.subject]}')
                       for d, color in zip(datasets, SUBJECT_COLORS)]
            ax.legend(handles=handles, fontsize=20, framealpha=0.9, loc='upper left')
        dress_panel(ax, dims, bounds, elev, azim)   # after the legend, so its text takes the house font

    fig.subplots_adjust(wspace=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=content_bbox(fig))

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'subjects',
        nargs   = '*',
        default = SUBJECTS,
        help    = 'subject names, each read from <data-dir>/<subject>/<subject>.json'
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
        default = OUTPUT,
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
        help    = 'number of Sobol points the front is evaluated on'
    )
    p.add_argument(
        '--seed',
        type    = int,
        default = SEED,
        help    = 'the random seed the Sobol points are generated with'
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
        subjects    = args.subjects,
        data_dir    = args.data_dir,
        path        = args.output,
        trial       = args.trial,
        scan        = args.scan,
        seed        = args.seed,
        dpi         = args.dpi,
        elev        = args.elev,
        azim        = args.azim,
        layout      = args.layout
    )
    print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
