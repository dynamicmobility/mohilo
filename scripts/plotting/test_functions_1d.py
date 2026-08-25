"""Plots every botorch synthetic test function this repo might optimize, at the
requested action dimension."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pypolar as plr

DIM   = 1     # 1 or 2
BOX   = 5.0   # half-width of the action box, where the function accepts one
N     = 2 ** 12
SEED  = 95
NCOLS = 4
PATH  = Path('scripts/output/synthetic_functions.svg')


def main():
    truths = {name: plr.construct_function(func, DIM, BOX, seed=SEED)
              for name, func in plr.SYNTHETIC_FUNCTIONS.items()}
    skipped = [name for name, truth in truths.items() if truth is None]
    truths  = {name: truth for name, truth in truths.items() if truth is not None}
    if skipped:
        print(f'no non-constant {DIM}D instance, skipped: {", ".join(skipped)}')

    nrows = -(-len(truths) // NCOLS)
    fig, axs = plt.subplots(
        nrows   = nrows,
        ncols   = NCOLS,
        figsize = (3 * NCOLS, 2.5 * nrows)
    )
    axs = np.atleast_1d(axs).ravel()

    for ax, (name, truth) in tqdm(zip(axs, truths.items()), total=len(truths)):
        # each function declares its own box, so the scan is per function
        X = plr.sample_actions(bounds=truth.bounds, n=N, kind='sobol', seed=SEED)
        plr.plot_test_function(ax, X, plr.truth_at(truth, X), name)

    for ax in axs[len(truths):]:
        ax.set_axis_off()

    fig.tight_layout()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PATH)
    print(f'wrote {PATH}')

if __name__ == '__main__':
    main()
