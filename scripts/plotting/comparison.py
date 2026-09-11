"""One metric against evaluation count, averaged over the trials in one dataset
directory.

One line per method with a +/-1 std band, so a method's spread across seeds is
visible next to its mean rather than hidden by it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypolar as plr

DATASET = Path('scripts/output/experiments/20260910_140158')
METRIC  = 'front_coverage'      # the IGD+ indicator (performance/mo.py)
COLORS  = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

# CSV column -> (axis label, output filename stem). The keys are the whitelist
# `--metric` accepts, so a mistyped column raises rather than plotting nothing.
METRICS = {
    'front_coverage'     : ('IGD+',                    'igd_plus'),
    'front_alignment'    : ('GD+',                     'gd_plus'),
    'hv_regret'          : ('HV regret (recommended)', 'hv_regret'),
    'hv_regret_attained' : ('HV regret (queried)',     'hv_regret_attained'),
}


def load(dataset: Path, metric: str = METRIC):
    """Every `*_trial*.csv` in `dataset`, grouped by the method its name states.

    Args:
        dataset: the dataset directory.
        metric: the column to read.

    Returns:
        {method: (evals, values)}, with `evals` (n_evals,) and `values`
        (n_trials, n_evals).
    """
    trials = {}
    for path in sorted(dataset.glob('*_trial*.csv')):
        method = path.stem.split('_trial')[0]
        frame  = pd.read_csv(path)
        trials.setdefault(method, []).append(
            (frame['evals'].to_numpy(), frame[metric].to_numpy())
        )

    runs = {}
    for method, records in trials.items():
        # a trial that stopped early would otherwise refuse to stack
        n = min(len(evals) for evals, _ in records)
        runs[method] = (records[0][0][:n],
                        np.vstack([values[:n] for _, values in records]))

    return runs


def plot_metric(ax, runs: dict, metric: str = METRIC, title: str = None):
    """Mean `metric` per method, with a +/-1 std band. Returns the `ax`."""
    label = METRICS[metric][0]
    for (method, (evals, values)), color in zip(runs.items(), COLORS):
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(evals, mean, color=color, lw=2,
                label=f'{method} (n={len(values)})')
        ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.2,
                        lw=0)

    ax.set_xlabel('evaluations')
    ax.set_ylabel(label)
    ax.set_title(title or label)
    ax.legend()
    plr.dress_axis(ax)
    return ax


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path, nargs='?', default=DATASET,
                   help='the dataset directory of *_trial*.csv runs to draw')
    p.add_argument('--metric', default=METRIC, choices=list(METRICS),
                   help=f'the column to draw (default {METRIC}, i.e. IGD+)')
    p.add_argument('--output', type=Path, default=None,
                   help="the figure to write; defaults to the dataset's own "
                        "path / <metric>.svg")

    return p.parse_args()


def main():
    args = parse_args()
    runs = load(args.dataset, args.metric)
    if not runs:
        raise SystemExit(f'no *_trial*.csv under {args.dataset}')

    label, stem = METRICS[args.metric]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_metric(ax, runs, args.metric,
                title=f'{label} on {args.dataset.name}')
    fig.tight_layout()
    output = args.output or args.dataset / f'{stem}.svg'
    fig.savefig(output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
