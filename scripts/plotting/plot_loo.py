"""Held-out R^2 against dataset size, for one recorded dataset.

Separates the two reasons a fit can be poor: a curve still climbing at the full
dataset means more measurements would help, while one that is flat and low means
the objective is noise-limited and more of the same will not.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from linear_operator.utils.warnings import NumericalWarning
from sklearn.metrics import r2_score

import pypolar as plr
from hilo.read_data import read_MH01_data, read_MT0x_data

warnings.filterwarnings('ignore', category=NumericalWarning)

DATASET         = 'MH01'    # 'MH01', or an MT0x CSV stem such as 'MT03_incline'
OBJECTIVES      = ['Metabolic Cost', 'Comfort Treadmill']   # MH01 only, None for all
SIZES           = None      # subset sizes, None for loo_curve's default of 5 through N
REPEATS         = 10        # subsets drawn per size
KIND            = 'random'  # 'random' (average case) or 'maximin' (space-filling)
SEED            = 95
GP_NOISE        = plr.NoiseModel.prior(0.3)
MIN_LENGTHSCALE = 0.1       # lengthscale floor, in the normalized action frame
OUTPUT          = Path('scripts/output/loo_curve.svg')


def fit_gp(objective, noise, hypers=None):
    """A GP on `objective`, its hyperparameters refit from that subset's points.

    `hypers` is accepted because `loo` passes it, and ignored: hyperparameters
    fitted on the full dataset would leak the held-out points into every subset
    and flatter the small-size end of the curve.
    """
    return plr.BoTorchGP(objective, noise=noise, fit_hyperparameters=True,
                         min_length_scale=MIN_LENGTHSCALE)


def load_objectives():
    """The objectives `DATASET` names, as a list."""
    if DATASET != 'MH01':
        return [read_MT0x_data(Path('human_data') / f'{DATASET}.csv', seed=SEED)]

    objectives = read_MH01_data()
    return [objectives[name] for name in (OBJECTIVES or objectives.names)]


def plot_loo(objectives, sizes=SIZES, repeats=REPEATS, kind=KIND):
    """Score curve per objective, one panel apiece.

    Args:
        objectives: the `Objective`s to score.
        sizes: subset sizes, or None for `loo_curve`'s default.
        repeats: subsets drawn per size.
        kind: 'random' or 'maximin' subsets.

    Returns:
        The figure that was drawn.
    """
    fig, axs = plt.subplots(
        ncols   = len(objectives),
        figsize = (4.5 * len(objectives), 3.6),
        squeeze = False
    )

    for ax, objective in zip(axs.ravel(), objectives):
        curve_sizes, scores = plr.loo_curve(
            objective, fit_gp, GP_NOISE, r2_score,
            sizes = sizes, repeats = repeats, kind = kind, seed = SEED
        )
        plr.plot_loo_curve(ax, curve_sizes, scores, ylabel='$R^2$',
                           title=objective.name)

        median = np.median(scores, axis=1)
        print(f'  {objective.name:<20}{median[0]:>+8.3f} at n={curve_sizes[0]}'
              f'{median[-1]:>+9.3f} at n={curve_sizes[-1]}')

    fig.tight_layout()
    return fig


def main():
    objectives = load_objectives()
    print(f'\n=== leave-one-out curve: {DATASET}, {KIND} subsets, {REPEATS} repeats ===')
    print(f'  {"objective":<20}{"first":>16}{"last":>17}')

    fig = plot_loo(objectives)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
