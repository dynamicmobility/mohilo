"""Validation of each subject's inferred Pareto front: at every validation
action, the GP's prediction joined to what was measured there, one subject per
row.

Each subject folder holds the optimization run `<subject>.json` and the
validation run `evaluation.json`, whose actions are marked 'pareto' or
'antipareto' by the source that chose them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import pypolar as plr

SUBJECTS    = [Path('human_data/MB02'), Path('human_data/MB04'), Path('human_data/MB05')]
OUTPUT      = Path('hilo/output/validation.jpg')

TRIAL       = -1        # the optimization trial whose GP is drawn
SCAN        = 2**14     # Sobol points the front is read off
SEED        = 95
DPI         = 300
LAYOUTS     = {'vertical': lambda n: (n, 1), 'horizontal': lambda n: (1, n)}  # (rows, cols) for n subjects

FRONT_COLOR   = '0.6'                                           # gray for the posterior cloud and front
SOURCE_COLORS = {'pareto': '#0072B2', 'antipareto': '#D55E00'}  # Okabe-Ito blue and vermillion
SOURCE_LABELS = {'pareto': 'Pareto', 'antipareto': 'anti-Pareto'}
MATCH_TOL     = 1e-4    # action distance within which a measurement belongs to a validation action


def front_order(mu, raw_mu):
    """Sorted nondominated indexes of the Pareto front."""
    nd_idx = plr.get_nondominated_tol(mu)

    return nd_idx[np.argsort(raw_mu[nd_idx, 0])]


def measured_at(
    objectives  : plr.DecoupledObjectives,
    actions     : np.ndarray,
    names       : list[str],
    tol         : float = MATCH_TOL
):
    """(n, m) mean measurement of each named objective at each of the (n, d)
    actions, averaging every repeat within `tol` of that action."""
    return np.column_stack([
        [objectives[name].ydata[np.linalg.norm(objectives[name].xdata - action, axis=1) < tol].mean()
         for action in actions]
        for name in names
    ])


def legend_handles():
    """Proxy artists: one line per validation source, one marker per value kind."""
    handles  = [Line2D([], [], color=color, lw=1.5, label=SOURCE_LABELS[source])
                for source, color in SOURCE_COLORS.items()]
    handles += [Line2D([], [], ls='', marker='*', ms=14, mfc='white', mec='black', label='GP prediction'),
                Line2D([], [], ls='', marker='s', ms=8, mfc='white', mec='black', label='measured')]
    return handles


def plot_validation(
    ax      : plt.Axes,
    run     : Path,
    trial   : int = TRIAL,
    scan    : int = SCAN,
    seed    : int = SEED
):
    """One subject's inferred front in gray, with each validation action's
    predicted value (star) joined to its mean measurement (square).

    Args:
        ax: the axes to draw on.
        run: the subject folder, holding `<subject>.json` and `evaluation.json`.
        trial: the optimization trial whose GP predicts.
        scan: Sobol points the front is read off.
        seed: seeds the Sobol scan.

    Returns:
        The ``ax`` that was drawn on.
    """
    dataset    = plr.ExperimentDataset.load(run / f'{run.name}.json')
    validation = plr.ExperimentDataset.load(run / 'evaluation.json')
    model      = dataset.get_model(trial)
    names      = model.objectives.names

    X          = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)
    mu, _      = model.posterior_at(X)
    raw_mu, _  = model.posterior_at(X, raw=True)
    plr.plot_pareto(
        ax                      = ax,
        pareto                  = raw_mu,
        nd_idx                  = front_order(mu, raw_mu),
        colors                  = FRONT_COLOR,
        connect                 = True,
        dominated_alpha         = 0.4,
        outline_nondominated    = 2,
        nondominated_s          = 30,
        label                   = 'inferred front',
        set_lims                = False
    )

    actions      = validation.get_actions()
    sources      = np.asarray(validation.get_sources())
    predicted, _ = model.posterior_at(actions, raw=True)
    measured     = measured_at(validation.get_objectives(), actions, names)
    for source, color in SOURCE_COLORS.items():
        rows = sources == source
        for start, end in zip(predicted[rows], measured[rows]):
            ax.plot(*np.stack([start, end]).T, color=color, lw=3, zorder=5)
        ax.scatter(*predicted[rows].T, marker='*', s=450, color=color,
                   edgecolor='black', lw=0.8, zorder=6)
        ax.scatter(*measured[rows].T, marker='s', s=100, color=color,
                   edgecolor='black', lw=0.8, zorder=6)

    ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    ax.set_ylabel(r'Comfort Score ($\uparrow$)')
    ax.set_title(f'Subject {dataset.subject}')

    return ax


def make_figure(
    runs    : list[Path],
    path    : Path = OUTPUT,
    trial   : int  = TRIAL,
    scan    : int  = SCAN,
    seed    : int  = SEED,
    dpi     : int  = DPI,
    layout  : str  = 'vertical',
    renames : list[str]  = None
):
    """Every subject's validation panel, one per row (`layout='vertical'`) or
    one per column (`'horizontal'`), written to `path`."""
    rows, cols = LAYOUTS[layout](len(runs))
    fig, axes  = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    if renames is None:
        renames = [None,] * len(runs)
    for i, (ax, run, rename) in enumerate(zip(axes.ravel(), runs, renames)):
        plot_validation(ax, run, trial, scan, seed)
        if i == 0:
            ax.legend(handles=legend_handles(), fontsize=10, framealpha=0.9)
        if rename: ax.set_title(f'Subject {''.join(rename)}')
        plr.dress_axis(
            ax, 
            label_size=22, 
            num_xticks=5, 
            num_yticks=6, 
            title_size=26
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'runs',
        type    = Path,
        nargs   = '*',
        default = SUBJECTS,
        help    = 'subject folders, each holding <subject>.json and evaluation.json'
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
        '--renames',
        type    = list[str],
        nargs   = '*',
        default = None,
        help    = 'subject moniker rename'
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
        '--layout',
        default = 'vertical',
        choices = list(LAYOUTS),
        help    = 'stack subjects one per row (vertical) or one per column (horizontal)'
    )

    return p.parse_args()


def main():
    args = parse_args()
    path = make_figure(
        runs    = args.runs,
        path    = args.output,
        trial   = args.trial,
        scan    = args.scan,
        seed    = args.seed,
        dpi     = args.dpi,
        layout  = args.layout,
        renames = args.renames
    )
    print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
