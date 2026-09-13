"""How far off each subject's GP was on the actions it queried: every query's
posterior prediction, from the GP fit before that feedback arrived, joined to
the feedback actually measured there, over the final GP's inferred front.

Each subject's optimization run is `<data-dir>/<subject>/<subject>.json`. Trial
k of a run holds the action queried at trial k and the GP fit after measuring
it, so trial k - 1's GP had not seen that feedback and its prediction is out of
sample. Trial 0's query has no earlier GP and is not drawn.

`--final-gp` predicts every query with the last trial's GP instead, which was
fit to all of this feedback, so its predictions are in-sample and trial 0's
query is drawn too. Those pairs are drawn in one color, with no query-trial
colorbar.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import pypolar as plr
from scripts.icra.validation_pareto import FRONT_COLOR, LAYOUTS, front_order, measured_at

SUBJECTS    = ['MB02', 'MB04', 'MB05']
DATA_DIR    = Path('human_data')
OUTPUT      = Path('hilo/output/feedback.jpg')

SCAN        = 2**14     # Sobol points the front is read off
SEED        = 95
DPI         = 300
CMAP        = 'viridis' # query trial, first to last
PAIR_COLOR  = '#0072B2' # every pair when they are not colored by trial (Okabe-Ito blue)


def query_trials(dataset: plr.ExperimentDataset, final_gp: bool = False):
    """(q,) every trial that queried an action and has a GP to predict it:
    all of them for the final GP, all but trial 0 otherwise."""
    first = 0 if final_gp else 1
    return np.array([k for k, trial in enumerate(dataset.trials)
                     if k >= first and trial.action is not None])


def query_pairs(dataset: plr.ExperimentDataset, names: list[str], final_gp: bool = False):
    """Each query's GP prediction and its mean measured feedback.

    Args:
        dataset: one subject's optimization run.
        names: objective names, in the column order to return.
        final_gp: predict with the last trial's GP instead of trial - 1's.

    Returns:
        trials: (q,) the trial each action was queried at.
        predicted: (q, m) raw posterior mean at the action, from trial - 1's GP
            or the final one.
        measured: (q, m) mean feedback at the action over every repeat.
        dropped: queries left out because an objective has no measurement.
    """
    trials    = query_trials(dataset, final_gp)
    actions   = np.array([dataset.trials[k].action for k in trials], dtype=float)
    models    = ([dataset.get_model(-1)] * len(trials) if final_gp
                 else [dataset.get_model(k - 1) for k in trials])
    predicted = np.vstack([model.posterior_at(action[None], raw=True)[0]
                           for model, action in zip(models, actions)])
    with warnings.catch_warnings():
        # an action with no rating averages an empty slice, which is NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        measured = measured_at(dataset.get_objectives(-1), actions, names)

    rated = ~np.isnan(measured).any(axis=1)
    return trials[rated], predicted[rated], measured[rated], int((~rated).sum())


def legend_handles(final_gp: bool = False):
    """Proxy artists for the front and the two value kinds."""
    prediction = 'final GP prediction' if final_gp else 'GP prediction before query'
    return [Line2D([], [], color=FRONT_COLOR, marker='o', mec='black', label='inferred front (final GP)'),
            Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec='black', label=prediction),
            Line2D([], [], ls='', marker='s', ms=8, mfc='white', mec='black', label='measured feedback')]


def plot_feedback(
    ax      : plt.Axes,
    dataset : plr.ExperimentDataset,
    norm    : Normalize | None,
    scan    : int = SCAN,
    seed    : int = SEED,
    final_gp: bool = False
):
    """One subject's final inferred front in gray, with each query's prediction
    (circle) joined to its measured feedback (square), colored by query trial.

    Args:
        ax: the axes to draw on.
        dataset: the subject's optimization run.
        norm: maps a query trial onto `CMAP`; None draws every pair in `PAIR_COLOR`.
        scan: Sobol points the front is read off.
        seed: seeds the Sobol scan.
        final_gp: predict with the last trial's GP instead of trial - 1's.

    Returns:
        The number of queries left out for a missing measurement.
    """
    model     = dataset.get_model(-1)
    names     = model.objectives.names
    X         = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)
    mu, _     = model.posterior_at(X)
    raw_mu, _ = model.posterior_at(X, raw=True)
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

    trials, predicted, measured, dropped = query_pairs(dataset, names, final_gp)
    colors = (matplotlib.colormaps[CMAP](norm(trials)) if norm is not None
              else np.tile(to_rgba(PAIR_COLOR), (len(trials), 1)))
    for start, end, color in zip(predicted, measured, colors):
        ax.plot(*np.stack([start, end]).T, color=color, lw=1.2, zorder=5)
    ax.scatter(*predicted.T, marker='o', s=60, c=colors, edgecolor='black', lw=0.6, zorder=6)
    ax.scatter(*measured.T, marker='s', s=60, c=colors, edgecolor='black', lw=0.6, zorder=6)

    ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    ax.set_ylabel(r'Comfort Score ($\uparrow$)')
    ax.set_title(f'Subject {dataset.subject}')

    return dropped


def make_figure(
    subjects    : list[str],
    data_dir    : Path = DATA_DIR,
    path        : Path = OUTPUT,
    scan        : int  = SCAN,
    seed        : int  = SEED,
    dpi         : int  = DPI,
    layout      : str  = 'vertical',
    final_gp    : bool = False
):
    """Every subject's feedback panel, one per row (`layout='vertical'`) or one
    per column (`'horizontal'`), sharing one query-trial colorbar, written to
    `path`. `final_gp` predicts every query with the last trial's GP."""
    datasets   = [plr.ExperimentDataset.load(data_dir / s / f'{s}.json') for s in subjects]
    queried    = [query_trials(d, final_gp) for d in datasets]
    norm       = (None if final_gp else
                  Normalize(min(q.min() for q in queried), max(q.max() for q in queried)))

    rows, cols = LAYOUTS[layout](len(datasets))
    fig, axes  = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False,
                              layout='constrained')
    for i, (ax, dataset, trials) in enumerate(zip(axes.ravel(), datasets, queried)):
        dropped = plot_feedback(ax, dataset, norm, scan, seed, final_gp)
        print(f'{dataset.subject}: {dropped} of {len(trials)} queries have a '
              'missing measurement and are not drawn')
        if i == 0:
            ax.legend(handles=legend_handles(final_gp), fontsize=10, framealpha=0.9)
        plr.dress_axis(ax, label_size=16, num_xticks=5, num_yticks=6, title_size=18)

    if norm is not None:
        cbar = fig.colorbar(ScalarMappable(norm, CMAP), ax=axes.ravel().tolist())
        cbar.set_label('query trial')
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # trials are whole numbers
        plr.dress_axis(cbar.ax, label_size=16)
        cbar.ax.grid(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')

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
    p.add_argument(
        '--final-gp',
        action  = 'store_true',
        help    = "predict every query with the last trial's GP (in-sample) instead of "
                  "the GP fit before that query's feedback"
    )

    return p.parse_args()


def main():
    args = parse_args()
    path = make_figure(
        subjects    = args.subjects,
        data_dir    = args.data_dir,
        path        = args.output,
        scan        = args.scan,
        seed        = args.seed,
        dpi         = args.dpi,
        layout      = args.layout,
        final_gp    = args.final_gp
    )
    print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
