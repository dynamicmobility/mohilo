"""One synthetic 1D run, drawn two ways: each objective's GP with the actions it
queried, and the Pareto front that GP found against the truth's own.

The run is a `scripts/icra/synthetic_1d.py` record, its groundtruth rebuilt from
the params it stored. The found front is the non-dominated set of the GP's
posterior mean over the truth's Sobol scan, the set the run's front metrics
score.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pypolar as plr
from pypolar.utils.plotting import FONT, MATH_FONT, SAMPLE_COLOR
from scripts.icra.validation_pareto import measured_at

TRIAL       = -1          # the trial whose GP is drawn
GRID_POINTS = 1024        # actions each GP panel is evaluated on
BAND_STD    = 2.0         # half-width of the GP band, in posterior standard deviations
NOISE_MARGIN = 2.0        # the front axes reach this many observation noise stds past the fronts
DPI         = 300
# Okabe-Ito, a published colorblind-safe palette
SURROGATE_COLOR  = '#D55E00'
TRUTH_COLOR  = '#000000'
SAMPLE_COLOR = '#009E73'
ARROWS      = {True: r'max', False: r'min'}   # the direction an objective improves in
OPT_COLORS  = ('blue', 'green')               # each objective's optimum star, in objective order
OPT_LABELS  = ('Met. opt.', 'Comf. opt.')
NEXT_COLOR  = 'red'                           # the next query's marker
STAR_S      = 500                             # optimum and next-query marker area, in points^2
MEASUREMENT_S = 70                            # measurement marker area, in points^2


def truth_columns(dataset: plr.ExperimentDataset, objectives: plr.DecoupledObjectives):
    """The truth's output column for each objective, in `objectives` order."""
    # return [dataset.groundtruth.objectives.index(name) for name in objectives.names]
    return [0, 1]


def true_optima(truth, objectives: plr.DecoupledObjectives, columns: list[int], n: int = GRID_POINTS):
    """Each objective's best action on an `n`-point grid over the truth's box.

    Returns:
        actions: (m, 1) row i is objective i's optimal action.
        values: (m, m) row i is every objective's noiseless truth at that action.
    """
    grid   = np.linspace(*truth.bounds[:, 0], n)[:, None]
    values = truth(grid, noise=False)[:, columns]
    best   = [np.argmax(o.ytransform(values[:, i])) for i, o in enumerate(objectives.objectives)]

    return grid[best], values[best]


def plot_fits(
    axes        : np.ndarray,
    dataset     : plr.ExperimentDataset,
    trial       : int   = TRIAL,
    n           : int   = GRID_POINTS,
    band_std    : float = BAND_STD
):
    """One panel per objective: its GP's posterior mean and band, the noiseless
    truth, and every measurement the GP was fit to, in raw units.

    Args:
        axes: one ``matplotlib.axes.Axes`` per objective, in the GP's order.
        dataset: the run to draw.
        trial: the trial whose GP is drawn.
        n: actions each panel is evaluated on.
        band_std: half-width of the band, in posterior standard deviations.

    Returns:
        The ``axes`` that were drawn on.
    """
    gp    = dataset.get_model(trial)
    truth = dataset.get_groundtruth()
    if gp.action_dim != 1:
        raise ValueError(f'each panel draws one action, got {gp.action_dim}D')

    grid    = np.linspace(*truth.bounds[:, 0], n)[:, None]
    mu, std = gp.posterior_at(grid, raw=True)
    values  = truth(grid, noise=False)[:, truth_columns(dataset, gp.objectives)]
    optimal_actions, optimal_values = true_optima(truth, gp.objectives, truth_columns(dataset, gp.objectives), n)
    for i, (ax, objective) in enumerate(zip(axes, gp.objectives.objectives)):
        plr.plot_fit_1d(
            ax       = ax,
            x        = grid[:, 0],
            mu       = mu[:, i],
            std      = std[:, i],
            xdata    = objective.xdata[:, 0],
            ydata    = objective.ydata,
            truth    = values[:, i],
            band_std = band_std,
            title    = f'{objective.name}',
            measurement_s=MEASUREMENT_S
        )
        ax.set_xlabel(r'Controller $\mathbf{a}$')
        ax.set_ylabel(f'{objective.name} ({ARROWS[objective.maximize]})')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.scatter(optimal_actions[i], optimal_values[i, i], s=STAR_S, c=OPT_COLORS[i], zorder=100, marker='*', label=OPT_LABELS[i])
        # if i == 0:
        #     ax.legend(frameon=True, fontsize=18)
        plr.dress_axis(
            ax,
            label_size=24,
            title_size=28
        )
    # every panel's entries, each label once
    entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    handles, labels = list(entries.values()), list(entries)
    axes[0].figure.legend(
        handles, labels,
        loc            = 'upper center',
        bbox_to_anchor = (0.5, 0.0),
        ncol           = int(np.ceil(len(labels) / 2)),   # two rows
        frameon        = True,
        prop           = {'family': FONT, 'math_fontfamily': MATH_FONT, 'size': 24},
    )

    return axes


def found_front(dataset: plr.ExperimentDataset, trial: int = TRIAL):
    """The truth's front and the front the GP found, both over the truth's scan.

    Returns:
        objectives: the GP's `DecoupledObjectives`, which fixes the column order.
        values: {'true front', 'predicted', 'true at found'} -> (k, m) raw values.
            'predicted' is the GP's mean at its found actions, 'true at found'
            the noiseless truth there, and 'true front' is sorted along its
            first column.
        noise_std: (m,) each objective's observation noise standard deviation.
    """
    gp         = dataset.get_model(trial)
    truth      = dataset.get_groundtruth()
    objectives = gp.objectives
    columns    = truth_columns(dataset, objectives)
    
    sorted_actions = truth.scan_actions[np.argsort(truth.scan_actions[:, 0])]
    print(sorted_actions.shape)
    sorted_truth   = truth(sorted_actions, noise=False)
    sorted_truth   = sorted_truth[:, columns]

    mu, _      = gp.posterior_at(sorted_actions, raw=True)
    found      = plr.get_nondominated(objectives.maximization_space(mu))
    true_front = sorted_truth[plr.get_nondominated(objectives.maximization_space(sorted_truth))]

    return objectives, {
        'all true'      : sorted_truth,
        'all pred'      : mu,
        'true front'    : true_front[np.argsort(true_front[:, 0])],
        'predicted'     : mu[found],
        # 'true at found' : scan[found],
    }, truth.noise_std[columns]


def plot_front(
    ax      : plt.Axes,
    dataset : plr.ExperimentDataset,
    trial   : int   = TRIAL,
    margin  : float = NOISE_MARGIN
):
    """The truth's front as a line, the found front at the GP's predicted and at
    its true values, and the measurements, in each objective's own units.

    The axes span every front widened by `margin` noise standard deviations per
    objective, and the legend counts the measurements that fall outside them.

    Returns:
        The ``ax`` that was drawn on.
    """
    objectives, values, noise_std = found_front(dataset, trial)
    actions  = np.unique(np.vstack([o.xdata for o in objectives.objectives]), axis=0)
    measured = measured_at(objectives, actions, objectives.names)

    fronts   = np.vstack(list(values.values()))
    lims     = np.stack([fronts.min(0) - margin * noise_std,
                         fronts.max(0) + margin * noise_std])
    outside  = np.any((measured < lims[0]) | (measured > lims[1]), axis=1).sum()
    ax.set_xlim(lims[:, 0])
    ax.set_ylim(lims[:, 1])

    # Measured points
    ax.scatter(
        *measured.T,
        s         = MEASUREMENT_S,
        color     = TRUTH_COLOR,
        zorder    = 8,
        label     = f'Measured feedback'
    )

    # Each objective's true optimum, at every objective's truth there
    truth   = dataset.get_groundtruth()
    columns = truth_columns(dataset, objectives)
    _, optimal_values = true_optima(truth, objectives, columns)
    for i, value in enumerate(optimal_values):
        ax.scatter(*value, s=STAR_S, c=OPT_COLORS[i], zorder=100, marker='*', label=OPT_LABELS[i])

    # The action this trial's GP chose next, at its noiseless truth
    action = dataset[trial].action
    if action is not None:
        ax.scatter(*truth(action[None, :], noise=False)[0, columns], s=STAR_S, c=NEXT_COLOR,
                   zorder=101, marker='X', label='Next query')

    # True landscape
    ax.plot(
        values['all true'][:, 0], 
        values['all true'][:, 1], 
        color   = TRUTH_COLOR,
        zorder  = 3,
        label   = 'True landscape',
        lw      = 2
    )

    # Surrogate landscape
    mu = values['all pred']
    ax.plot(
        mu[:,0], 
        mu[:, 1], 
        color   = SURROGATE_COLOR,
        label   = 'Surrogate landscape',
        lw      = 2,
        ls      = '--',
        zorder  = 4
    )
    

    # ax.legend(fontsize=8, loc='upper right', framealpha=0.9, markerscale=0.4)
    ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    ax.set_ylabel(r'Comfort ($\uparrow$)')

    ax.set_xticks([])
    ax.set_yticks([])

    plr.dress_axis(
        ax,
        label_size=24,
        title_size=28
    )

    return ax


def make_figures(
    dataset : plr.ExperimentDataset,
    output  : Path = None,
    trial   : int  = TRIAL,
    dpi     : int  = DPI
):
    """Both figures for one run, written to `output`, or beside the run.

    Returns:
        The paths of the GP figure and of the front figure.
    """
    output = Path(output or dataset.path.parent)
    output.mkdir(parents=True, exist_ok=True)

    m         = len(dataset.groundtruth.objectives)
    fig, axes = plt.subplots(1, m, figsize=(6 * m, 4.0), squeeze=False)
    plot_fits(axes[0], dataset, trial)
    fig.tight_layout()
    fits = output / f'{dataset.path.stem}_gp_fits.svg'
    fig.savefig(fits, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_front(ax, dataset, trial)
    fig.tight_layout()
    front = output / f'{dataset.path.stem}_pareto_front.svg'
    fig.savefig(front, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)

    return fits, front


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path,
                   help='the run to draw, as written by scripts/icra/synthetic_1d.py')
    p.add_argument('--trial', type=int, default=TRIAL,
                   help='the trial whose GP is drawn')
    p.add_argument('--output', type=Path, default=None,
                   help="the directory both figures are written to; defaults to the run's own")

    return p.parse_args()


def main():
    args    = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset)
    for path in make_figures(dataset, args.output, args.trial):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
