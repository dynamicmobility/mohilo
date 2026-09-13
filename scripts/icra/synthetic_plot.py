"""The synthetic MO-GP vs NSGA-2 comparison, drawn two ways: every method's
metric curve at every noise level on one axis, and the Pareto front each method
holds after its last evaluation, one panel per noise level.

Each run directory is one `scripts/comparison.py` output: `*_trial*.csv`
metric curves, `{seed}.json` MO-GP records and `NSGA2_trial{i}.npz` NSGA-2
queries.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypolar as plr
import hilo.shared.simulation as hilo
from scripts.plotting.comparison import METRIC, METRICS, load

SYNTHETIC_RUNS = [
    Path('scripts/output/experiments/20260912_125617'),
    Path('scripts/output/experiments/20260912_131239'),
    Path('scripts/output/experiments/20260912_133348'),
]
OUTPUT = Path('scripts/output/experiments')
TRIAL  = 0                                   # trial index the Pareto panels draw
CMAPS  = {'MOBO': 'Blues', 'NSGA2': 'Reds'}  # one hue per method
SHADES = (0.45, 0.95)                        # colormap span, lowest to highest noise
FRONT_SAMPLES = 2 ** 20                     # Sobol actions the true front is read off


def noise_level(run: Path):
    """The truth's `rel_noise_std`, read off any MO-GP record in `run`."""
    record = next(run.glob('*.json'), None)
    if record is None:
        raise SystemExit(f'no MO-GP .json record under {run}')

    return json.loads(record.read_text())['groundtruth']['rel_noise_std']


def plot_overlay(ax, runs: list[Path], metric: str = METRIC):
    """Mean `metric` per method and noise level, with a +/-1 std band. A method
    keeps one hue and darkens with noise. Returns the `ax`."""
    runs = sorted(runs, key=noise_level)
    for shade, run in zip(np.linspace(*SHADES, len(runs)), runs):
        for method, (evals, values) in load(run, metric).items():
            color     = plt.get_cmap(CMAPS[method])(shade)
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(evals, mean, color=color, lw=2,
                    label=f'{method}, noise {noise_level(run):g} (n={len(values)})')
            ax.fill_between(evals, mean - std, mean + std, color=color,
                            alpha=0.12, lw=0)

    label = METRICS[metric][0]
    ax.set_xlabel('evaluations')
    ax.set_ylabel(label)
    ax.set_title(f'{label} by noise level')
    ax.legend(fontsize=8)
    return plr.dress_axis(ax)


def dense_front(
    truth       : plr.MOSyntheticOracle,
    objectives  : plr.DecoupledObjectives,
    n           : int = FRONT_SAMPLES
):
    """(k, m) the truth's front over an `n`-point Sobol scan of its box, sorted
    by the first objective so it draws as a line."""
    actions = plr.sample_actions(truth.bounds, n=n, kind='sobol', seed=0)
    values  = truth(actions, noise=False)
    front   = values[plr.get_nondominated(objectives.maximization_space(values))]

    return front[np.argsort(front[:, 0])]


def fronts(run: Path, trial: int = TRIAL):
    """Both methods' final Pareto sets in `run`, at their noiseless true values.

    MO-GP's set is the nondominated scan actions of its last posterior mean;
    NSGA-2's is the nondominated actions among its noisy measurements.

    Returns:
        dataset: the MO-GP `ExperimentDataset`.
        truth: the `MOSyntheticOracle` both were scored against.
        values: {'true front', 'MOBO', 'NSGA2', 'MOBO last query'} -> (k, m).
    """
    npz = run / f'NSGA2_trial{trial}.npz'
    if not npz.exists():
        raise SystemExit(f'{npz} is missing; rerun scripts/comparison.py, which '
                         'saves NSGA-2 queries')

    dataset    = plr.ExperimentDataset.load(run / f'{hilo.SEED + trial}.json')
    truth      = dataset.get_groundtruth()
    gp         = dataset.get_model()
    objectives = gp.objectives
    nsga2      = np.load(npz)

    # posterior_at defaults to maximization space, what get_nondominated takes
    mu, _      = gp.posterior_at(truth.scan_actions)
    nsga2_set  = plr.get_nondominated(objectives.maximization_space(nsga2['measured']))

    return dataset, truth, {
        'true front'      : dense_front(truth, objectives),
        'MOBO'            : truth.scan_values[plr.get_nondominated(mu)],
        'NSGA2'           : truth(nsga2['queried'][nsga2_set], noise=False),
        'MOBO last query' : truth(dataset.get_actions()[-1:], noise=False),
    }


def plot_fronts(ax, run: Path, trial: int = TRIAL):
    """The true front, each method's final front, and MO-GP's last query, in
    the truth's own units. Returns the `ax`."""
    dataset, truth, values = fronts(run, trial)
    for method in ('MOBO', 'NSGA2'):
        front = values[method][np.argsort(values[method][:, 0])]
        ax.plot(*front.T, marker='o', ms=4, lw=1.2, alpha=0.8, zorder=4,
                color=plt.get_cmap(CMAPS[method])(SHADES[1]),
                label=f'{method} front')

    maximize = dataset.get_objectives().maximize
    return plr.plot_mo_space(
        ax,
        true_front = values['true front'],
        overlays   = {'MOBO last query': values['MOBO last query']},
        names      = [f'{name} ({"higher" if up else "lower"} is better)'
                      for name, up in zip(('$f_1$', '$f_2$'), maximize)],
        title      = f'noise {truth.rel_noise_std:g}, seed {hilo.SEED + trial}, '
                     f'{len(dataset)} evaluations'
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('runs', type=Path, nargs='*', default=SYNTHETIC_RUNS,
                   help='run directories, one per noise level')
    p.add_argument('--metric', default=METRIC, choices=list(METRICS),
                   help=f'the curve to overlay (default {METRIC}, i.e. IGD+)')
    p.add_argument('--trial', type=int, default=TRIAL,
                   help='the trial index the Pareto panels draw')
    p.add_argument('--output', type=Path, default=OUTPUT,
                   help='the directory both figures are written to')

    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_overlay(ax, args.runs, args.metric)
    fig.tight_layout()
    output = args.output / f'{METRICS[args.metric][1]}_by_noise.svg'
    fig.savefig(output)
    print(f'wrote {output}')

    runs = sorted(args.runs, key=noise_level)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 5),
                             squeeze=False)
    for ax, run in zip(axes[0], runs):
        plot_fronts(ax, run, args.trial)
    fig.tight_layout()
    output = args.output / f'pareto_trial{args.trial}.svg'
    fig.savefig(output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
