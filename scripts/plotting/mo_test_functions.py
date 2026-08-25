"""Plots the objective space of every two-objective botorch test function this
repo might optimize: a Sobol scan of the whole action box, with the
non-dominated front picked out.

Direction is explicit. Every botorch test problem is a *minimization* at
`negate=False`, which is the form `Objective` expects since it encodes direction
itself through `maximize`. `MAXIMIZE` states that per objective and is applied
the way `Objective` applies it -- multiply by `+1` to maximize and `-1` to
minimize, which turns the values larger-is-better, the frame
`get_nondominated` reads. The axis labels carry the arrow so a front is never
read in the wrong direction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pypolar as plr

NUM_OBJECTIVES = 2       # the registry entries plotted; the rest are skipped
MAXIMIZE       = (True, True)  # direction per objective, as `Objective` takes it
DIMS           = range(1, 8)     # action dims tried, smallest first
BOX            = 5.0     # half-width where a function takes one; MO problems do not
N_SCAN         = 2 ** 14 # Sobol points of the action box
N_FRONT        = 512     # points of the analytic front, where botorch has one
SHOW_TRUE_FRONT = True   # overlay it, so a scan that never reaches it is visible
SEED           = 95
NCOLS          = 4
PATH           = Path('scripts/output/mo_synthetic_functions.svg')

ARROWS = {True: '↑', False: '↓'}


def build(func, seed=SEED):
    """The instance of func with NUM_OBJECTIVES objectives at the smallest
    action dimension it has one, or None."""
    for dim in DIMS:
        truth = plr.construct_function(func, dim, BOX, num_objectives=NUM_OBJECTIVES,
                                       seed=seed)
        if truth is not None and truth.num_objectives == NUM_OBJECTIVES:
            return truth

    return None


def true_front(truth, n=N_FRONT):
    """The analytic front, for the 7 of 16 entries botorch implements one for.
    The rest inherit a `gen_pareto_front` that raises."""
    try:
        return truth.gen_pareto_front(n).numpy()
    except NotImplementedError:
        return None


def plot_front(ax, name, truth, seed=SEED):
    """One function's objective space: the scanned box, and its front in red."""
    X = plr.sample_actions(bounds=truth.bounds, n=N_SCAN, kind='sobol', seed=seed)
    Y = plr.truth_at(truth, X)

    # sign * Y is larger-is-better, which is the frame get_nondominated reads;
    # the plot stays in the function's own units
    sign  = np.where(MAXIMIZE, 1.0, -1.0)
    front = plr.get_nondominated(sign * Y)

    # rasterized: N_SCAN points per function is megabytes of SVG as vectors
    ax.scatter(Y[:, 0], Y[:, 1], s=3, c='0.85', lw=0, label='box', rasterized=True)
    ax.scatter(Y[front, 0], Y[front, 1], s=6, c='crimson', lw=0, label='pareto',
               rasterized=True)

    # scattered rather than joined: ZDT3's front is disconnected, and a line
    # would bridge its segments
    P = true_front(truth) if SHOW_TRUE_FRONT else None
    if P is not None:
        ax.scatter(P[:, 0], P[:, 1], s=3, c='k', lw=0, label='true front',
                   rasterized=True)

    ax.set(xlabel=f'f1 {ARROWS[MAXIMIZE[0]]}', ylabel=f'f2 {ARROWS[MAXIMIZE[1]]}',
           title=f'{name}  (d={truth.dim})')
    ax.title.set_fontsize('medium')


def main():
    truths  = {name: build(func) for name, func in plr.MO_SYNTHETIC_FUNCTIONS.items()}
    skipped = [name for name, truth in truths.items() if truth is None]
    truths  = {name: truth for name, truth in truths.items() if truth is not None}
    if skipped:
        print(f'no {NUM_OBJECTIVES}-objective instance, skipped: {", ".join(skipped)}')

    direction = ', '.join(f'f{j + 1} {ARROWS[m]}' for j, m in enumerate(MAXIMIZE))
    print(f'{len(truths)} functions, {direction}')

    nrows = -(-len(truths) // NCOLS)
    fig, axs = plt.subplots(
        nrows   = nrows,
        ncols   = NCOLS,
        figsize = (3.2 * NCOLS, 2.9 * nrows),
        layout  = 'constrained'
    )
    axs = np.atleast_1d(axs).ravel()

    for ax, (name, truth) in tqdm(zip(axs, truths.items()), total=len(truths)):
        plot_front(ax, name, truth)

    for ax in axs[len(truths):]:
        ax.set_axis_off()

    # gathered across axes, since only some entries carry a true front
    seen = {}
    for ax in axs[:len(truths)]:
        seen.update({l: h for h, l in zip(*ax.get_legend_handles_labels())
                     if l not in seen})
    fig.legend(seen.values(), seen.keys(), loc='outside lower center',
               ncols=len(seen), markerscale=3, frameon=False)

    PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PATH)
    print(f'wrote {PATH}')


if __name__ == '__main__':
    main()
