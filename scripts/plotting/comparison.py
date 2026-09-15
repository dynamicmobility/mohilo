"""One metric against evaluation count, averaged over the trials in one dataset
directory, or over every condition of an ablation directory on one axis.

One line per method and condition with a +/-1 std band, so a method's spread
across seeds is visible next to its mean rather than hidden by it. An ablation
directory holds one subdirectory of `*_trial*.csv` runs per condition, e.g.
`noise0.1/`. Each method keeps one hue, darker for a larger number in the
condition's name.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatter
import numpy as np
import pandas as pd
import pypolar as plr
from pypolar.utils.plotting import FONT, MUTED, TICK_SIZE
import hilo.shared.simulation as hilo

DATASET = Path('scripts/output/experiments/20260910_140158')
METRIC  = 'front_coverage'      # the IGD+ indicator (performance/mo.py)
SKIP    = hilo.NUM_RANDOM       # rows at or below this many evaluations are dropped
NSGA    = 'NSGA2'               # the method name `NSGA2_trial*.csv` files carry

# (light, mid, dark) per method, in sorted method order; mid is the categorical color
RAMPS   = [('#86b6ef', '#2a78d6', '#104281'),   # blue
           ('#ff9067', '#eb6834', '#752600'),   # orange
           ('#42c891', '#1baf7a', '#015136')]   # aqua

# CSV column -> (axis label, output filename stem). The keys are the whitelist
# `--metric` accepts, so a mistyped column raises rather than plotting nothing.
METRICS = {
    'front_coverage'     : ('IGD+',                    'igd_plus'),
    'front_alignment'    : ('GD+',                     'gd_plus'),
    'hv_regret'          : ('HV regret (recommended)', 'hv_regret'),
    'hv_regret_attained' : ('HV regret (queried)',     'hv_regret_attained'),
}


def load(dataset: Path, metric: str = METRIC, skip: int = SKIP):
    """Every `*_trial*.csv` in `dataset`, grouped by the method its name states.

    Rows at `skip` evaluations or fewer are dropped. Over that few points the
    posterior is near flat, so nearly every scanned action is nondominated and
    the metrics score a front the model never found.

    Args:
        dataset: the dataset directory.
        metric: the column to read.
        skip: the evaluation count at or below which rows are dropped.

    Returns:
        {method: (evals, values)}, with `evals` (n_evals,) and `values`
        (n_trials, n_evals).
    """
    trials = {}
    for path in sorted(dataset.glob('*_trial*.csv')):
        method = path.stem.split('_trial')[0]
        frame  = pd.read_csv(path)
        frame  = frame[frame['evals'] > skip]
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


def condition_value(path: Path):
    """The last number in a condition directory's name, e.g. 0.1 for `noise0.1`."""
    numbers = re.findall(r'\d+(?:\.\d+)?', path.name)
    return float(numbers[-1]) if numbers else float('inf')


def conditions(dataset: Path):
    """The directories of `*_trial*.csv` runs: `dataset` itself if it holds
    any, otherwise each subdirectory that does, by increasing `condition_value`."""
    if any(dataset.glob('*_trial*.csv')):
        return [dataset]

    return sorted((d for d in dataset.iterdir()
                   if d.is_dir() and any(d.glob('*_trial*.csv'))),
                  key=lambda d: (condition_value(d), d.name))


def truncate(evals, values, max_evals: int, name: str):
    """The rows at `max_evals` evaluations or fewer, as `(evals, values)`.

    Warns when `max_evals` is not itself a recorded evaluation count, since the
    curve then stops at the last count below it rather than at `max_evals`.
    """
    keep = evals <= max_evals
    if not keep.any():
        raise SystemExit(f'{name} starts at {evals[0]} evaluations, past '
                         f'--max-nsga-queries {max_evals}')

    if max_evals not in evals:
        above   = evals[~keep]
        nearest = (f'the nearest recorded are {evals[keep][-1]} and {above[0]}'
                   if len(above) else f'the last recorded is {evals[-1]}')
        print(f'warning: {name} has no row at exactly {max_evals} evaluations; '
              f'{nearest}. Truncating at {evals[keep][-1]}.')

    return evals[keep], values[:, keep]


def shades(ramp, n: int):
    """n colors from the light to the dark end of one method's ramp, or its mid
    step when n = 1."""
    cmap = LinearSegmentedColormap.from_list('ramp', ramp)
    return [cmap(t) for t in (np.linspace(0, 1, n) if n > 1 else [0.5])]


def plot_run(ax, evals, values, color, label: str):
    """Mean over trials with a +/-1 std band; `values` is (n_trials, n_evals).
    Returns the `ax`."""
    mean, std = values.mean(axis=0), values.std(axis=0)
    ax.plot(evals, mean, color=color, lw=2, label=f'{label} (n={len(values)})')
    ax.fill_between(evals, mean - std, mean + std, color=color, alpha=0.12,
                    lw=0)
    return ax


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path, nargs='?', default=DATASET,
                   help='a dataset directory of *_trial*.csv runs, or an '
                        'ablation directory of such directories')
    p.add_argument('--metric', default=METRIC, choices=list(METRICS),
                   help=f'the column to draw (default {METRIC}, i.e. IGD+)')
    p.add_argument('--output', type=Path, default=None,
                   help="the figure to write; defaults to the dataset's own "
                        "path / <metric>.svg")
    p.add_argument('--logx', action='store_true',
                   help='log-scale the evaluations axis')
    p.add_argument('--max-nsga-queries', type=int, default=None,
                   help=f'drop {NSGA} rows past this many evaluations; warns '
                        'unless it is one of the recorded counts')
    p.add_argument('--exclude', nargs='+', default=[], metavar='METHOD',
                   help='methods not to draw, e.g. --exclude MOBO-sparse; the '
                        'rest keep their colors')

    return p.parse_args()


def main():
    args = parse_args()
    dirs = conditions(args.dataset)
    if not dirs:
        raise SystemExit(f'no *_trial*.csv under {args.dataset} or its '
                         'subdirectories')

    label, stem = METRICS[args.metric]
    runs    = {run_dir: load(run_dir, args.metric) for run_dir in dirs}
    if args.max_nsga_queries is not None:
        for run_dir, run in runs.items():
            if NSGA in run:
                run[NSGA] = truncate(*run[NSGA], args.max_nsga_queries,
                                     f'{run_dir.name}/{NSGA}')
    methods = sorted({method for run in runs.values() for method in run})
    if len(methods) > len(RAMPS):
        raise SystemExit(f'{len(methods)} methods {methods}, but only '
                         f'{len(RAMPS)} color ramps')

    unknown = set(args.exclude) - set(methods)
    if unknown:
        raise SystemExit(f'--exclude {sorted(unknown)} not among the methods '
                         f'{methods}')
    shown = [method for method in methods if method not in args.exclude]
    if not shown:
        raise SystemExit(f'--exclude removes every method {methods}')

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, ramp in zip(methods, RAMPS):
        if method not in shown:
            continue
        for run_dir, color in zip(dirs, shades(ramp, len(dirs))):
            if method not in runs[run_dir]:
                continue
            name = method if len(dirs) == 1 else f'{method}, {run_dir.name}'
            plot_run(ax, *runs[run_dir][method], color, name)

    ax.set_xlabel('evaluations')
    ax.set_ylabel(label)
    ax.set_title(f'{label} on {args.dataset.name}')
    # ax.legend(ncols=len(shown), fontsize='small', loc='upper center',bbox_to_anchor=(0.5, -0.15))
    plr.dress_axis(ax)
    if args.logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(LogFormatter())
        ax.xaxis.set_minor_formatter(LogFormatter())
        ax.tick_params(axis='x', which='minor', colors=MUTED,
                       labelsize=TICK_SIZE, labelfontfamily=FONT)
    output = args.output or args.dataset / f'{stem}.svg'
    fig.savefig(output, bbox_inches='tight')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
