"""IGD+ against query count for MO-HILBO and NSGA-II, at each action dimension.

`figure` is written to take the condition prefix and label symbol, so
`final_obj.py` draws the objective-count ablation from the same code.

One line per method and condition with a +/-1 std band over the trials, taking
the colors of `scripts/plotting/comparison.py`: one hue per method, darker for
a larger condition, and NSGA-II dashed. NSGA-II records past `MAX_QUERIES` are
dropped, so both methods are read at the same budget. No legend is drawn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pypolar as plr

DATASET     = Path('scripts/output/final_exp/dim-100-again')
OUTPUT      = DATASET / 'final_igd_plus.svg'
PREFIX      = 'dim'             # the condition subdirectory prefix
SYMBOL      = 'n'               # the label symbol the condition is named by
METRIC      = 'front_coverage'  # the IGD+ indicator (performance/mo.py)
SKIP        = 3                 # rows at or below this many evaluations are dropped
MAX_QUERIES = 100               # every method is truncated at this many evaluations

# method -> (label prefix, (light, mid, dark) ramp, linestyle), in the colors
# comparison.py gives them: blue is the first of its sorted methods and aqua the
# third, so excluding MOBO-sparse leaves the orange ramp unused. The labels name
# the lines for a legend drawn by hand; the figure carries none.
METHODS = {
    'MOBO-dense' : ('MO-HILBO', ('#86b6ef', '#2a78d6', '#104281'), '-'),
    'NSGA2'      : ('NSGA-II',  ('#42c891', '#1baf7a', '#015136'), '--'),
}

TICK_SIZE   = 22
LABEL_SIZE  = 28


def load(run_dir: Path, method: str):
    """Every `<method>_trial*.csv` in `run_dir`, as `(evals, values)` with
    `evals` (n_evals,) and `values` (n_trials, n_evals).

    Rows at `SKIP` evaluations or fewer and past `MAX_QUERIES` are dropped.
    """
    records = []
    for path in sorted(run_dir.glob(f'{method}_trial*.csv')):
        frame = pd.read_csv(path)
        frame = frame[(frame['evals'] > SKIP) & (frame['evals'] <= MAX_QUERIES)]
        records.append((frame['evals'].to_numpy(), frame[METRIC].to_numpy()))

    # a trial that stopped early would otherwise refuse to stack
    n = min(len(evals) for evals, _ in records)
    return records[0][0][:n], np.vstack([values[:n] for _, values in records])


def conditions(dataset: Path, prefix: str):
    """The `<prefix><k>` subdirectories holding runs, as `(k, path)` by
    increasing k."""
    dirs = [(int(d.name[len(prefix):]), d) for d in dataset.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
            and any(d.glob('*_trial*.csv'))]
    return sorted(dirs)


def shades(ramp, n: int):
    """n colors from the light to the dark end of one method's ramp, or its mid
    step when n = 1."""
    cmap = LinearSegmentedColormap.from_list('ramp', ramp)
    return [cmap(t) for t in (np.linspace(0, 1, n) if n > 1 else [0.5])]


def figure(dataset: Path, prefix: str, symbol: str, output: Path, ylim=None):
    """Draw and write the figure for one ablation directory.

    Args:
        dataset: the ablation directory of `<prefix><k>/` run directories.
        prefix: the condition subdirectory prefix, e.g. `dim`.
        symbol: the label symbol the condition is named by, e.g. `n`.
        output: the figure to write.
    """
    runs = conditions(dataset, prefix)
    if not runs:
        raise SystemExit(f'no {prefix}<k>/ runs under {dataset}')

    fig, ax = plt.subplots(figsize=(11, 7))
    for method, (name, ramp, style) in METHODS.items():
        for (k, run_dir), color in zip(runs, shades(ramp, len(runs))):
            evals, values = load(run_dir, method)
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(evals, mean, color=color, lw=3, ls=style,
                    label=f'{name} (${symbol}={k}$)')
            ax.fill_between(evals, mean - std, mean + std, color=color,
                            alpha=0.12, lw=0)

    ax.set_xlabel('Queries')
    ax.set_ylabel('IGD+')
    ax.set_ylim(ylim)
    plr.dress_axis(ax, tick_size=TICK_SIZE, label_size=LABEL_SIZE)
    fig.savefig(output, bbox_inches='tight')
    print(f'wrote {output}')


def main():
    figure(DATASET, PREFIX, SYMBOL, OUTPUT, ylim=(-0.05, 0.3))


if __name__ == '__main__':
    main()
