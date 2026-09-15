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
from pypolar.utils.plotting import SAMPLE_COLOR
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
ARROWS      = {True: r'$\uparrow$', False: r'$\downarrow$'}   # the direction an objective improves in


def truth_columns(dataset: plr.ExperimentDataset, objectives: plr.DecoupledObjectives):
    """The truth's output column for each objective, in `objectives` order."""
    return [dataset.groundtruth.objectives.index(name) for name in objectives.names]


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
            title    = f'{objective.name}, {len(objective.ydata)} measurements'
        )
        ax.set_xlabel(dataset.get_action_labels()[0])
        ax.set_ylabel(f'{objective.name} ({ARROWS[objective.maximize]})')
        plr.dress_axis(ax)

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

    # Measured points (x)
    ax.scatter(
        *measured.T, 
        s         = 80,
        marker    = 'x',
        lw        = 1.0,
        color     = SAMPLE_COLOR,
        alpha     = 0.7,
        zorder    = 6,
        label     = f'Measured feedback'
    )

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
    

    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    ax = plr.dress_axis(ax)

    

    ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    ax.set_ylabel(r'Comfort ($\uparrow$)')
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
    fig, axes = plt.subplots(1, m, figsize=(6 * m, 4.5), squeeze=False)
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
