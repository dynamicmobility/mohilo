"""Leave-one-out cross-validation and R^2 for one GP on a synthetic objective.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions import (
    Ackley,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
)
from sklearn.metrics import r2_score
import pypolar as plr
import pandas as pd
from hilo.read_data import read_MH01_data, read_MT0x_data

DIM          = 3        # action dimension
NUM_SAMPLES  = 15       # measurements drawn from the objective
NOISE        = 0.5      # noise added, as a fraction of the truth's spread
GP_NOISE     = plr.NoiseModel.prior(1.0)  # a float pins, None fits, prior() regularizes
# GP_NOISE     = plr.NoiseModel.fitted()  # a float pins, None fits, prior() regularizes
MIN_LENGTHSCALE = 0.1   # lengthscale floor, at the design spacing
REFIT_FOLDS  = False    # refit kernel hyperparameters inside every LOO fold
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
DESIGN       = 'sobol'  # 'sobol' (space-filling) or 'uniform' (iid)
SEED         = 95
OUTPUT       = Path('hilo/output/gp_diagnostics.png')


LENGTHSCALE = 0.2
SIGNAL_VAR = 1.0
NOISE_VAR = 0.3 ** 2
HYPERS = {
    'lengthscale': LENGTHSCALE,
    'signal_var': SIGNAL_VAR,
    'noise_var': NOISE_VAR
}
HYPERS = None


PLOT_RES     = 400      # plotting only
SPREAD_REF   = 4096     # samples used to measure the truth's spread

# Okabe-Ito, a published colorblind-safe palette
TRUTH_COLOR = '#000000'
FOLD_COLOR  = '#56B4E9'
FULL_COLOR  = '#D55E00'
POINT_COLOR = '#0072B2'


def fit_gp(objective, noise, hypers=None):
    """A GP on `objective`, with its hyperparameters refit or frozen.

    Args:
        objective: the measurements to condition on.
        noise: a `NoiseModel`, or a fraction of the objective's spread to pin,
            or None to fit it by marginal likelihood alongside the kernel.
        hypers: a `get_fitted_hyperparameters()` dict to hold fixed, or None to
            fit them. Every entry supersedes the arguments, the noise included,
            so a fitted noise freezes across folds exactly as the kernel does.
    """
    if hypers is None:
        return plr.BoTorchGP(objective, noise=noise, fit_hyperparameters=True,
                             min_length_scale=MIN_LENGTHSCALE)

    # Standardize divides train_Y by its own sample spread, so the square root
    # of a post-Standardize noise variance is the fraction a pinned noise states
    return plr.BoTorchGP(
        objective,
        noise               = plr.NoiseModel.pinned(np.sqrt(hypers['noise_var'])),
        fit_hyperparameters = False,
        length_scale        = hypers['lengthscale'],
        signal_var          = hypers['signal_var'],
        min_length_scale    = MIN_LENGTHSCALE
    )


def leave_one_out(objective, noise_std, hypers):
    """Held-out posterior at every measured action.

    Fold i is refit on every measurement except i, and is then asked to predict 
    point i.

    Args:
        objective: the full set of measurements.
        noise_std: what each fold's GP assumes, a fraction of its own spread.
        hypers: a `get_fitted_hyperparameters()` dict frozen in every fold, or
            None to refit them fold by fold.

    Returns:
        (mu, std, models): mu and std are the predicted (held-out) point for each
        respective fold in raw unit. Models is the GP of each fold.
    """
    n = objective.ydata.size
    mu, std, models = np.empty(n), np.empty(n), []

    for i in range(n):
        keep = np.delete(np.arange(n), i)
        fold = plr.Objective.from_data(
            actions  = objective.xdata[keep],
            values   = objective.ydata[keep],
            maximize = objective.maximize,
            name     = objective.name
        )
        gp = fit_gp(fold, noise_std, hypers)
        fold_mu, fold_std = gp.posterior_at(objective.xdata[i], raw=True)

        mu[i], std[i] = fold_mu[0, 0], fold_std[0, 0]
        models.append(gp)

    return mu, std, models


def main():
    torch.manual_seed(SEED)

    bounds = torch.tensor([[-BOX] * DIM, [BOX] * DIM], dtype=torch.float64)
    if False:
        sampled_actions = plr.sample_actions(
            bounds = bounds,
            n      = NUM_SAMPLES,
            kind   = 'sobol',
            seed   = SEED
        )
        objective = plr.Objective.from_synthetic(
            function      = Levy,
            actions       = sampled_actions,
            maximize      = False,
            rel_noise_std = NOISE,
            seed          = SEED,
            name          = 'Synthetic Objective'
        )
    else:
        objectives = read_MH01_data(
            objs_path    = Path('human_data/pilot_mohilo.csv'),
            actions_path = Path('human_data/MH01_walk.csv')
        )
        objective = read_MT0x_data(
            Path('human_data/MT03_incline.csv'),
            # Path('human_data/MT03_incline_outliers.csv'),
            # Path('human_data/MT04_incline.csv'),
            # Path('human_data/MT05_incline.csv'),
            # num_samples=NUM_SAMPLES,
            seed=92
        )
        objective: plr.Objective = objectives['Metabolic Cost']

    full_gp = fit_gp(objective, GP_NOISE, hypers=None)
    hypers  = HYPERS if REFIT_FOLDS else full_gp.get_fitted_hyperparameters()

    fitted = full_gp.get_fitted_hyperparameters()
    print(f'full-data fit: noise_std {np.sqrt(fitted["noise_var"]):.3f} of spread, '
          f'lengthscales {np.round(fitted["lengthscale"], 2)}')

    loo_mu, loo_std, models = leave_one_out(objective, GP_NOISE, hypers)
    in_sample_mu, _ = full_gp.posterior_at(objective.xdata, raw=True)

    r2 = r2_score(objective.ydata, loo_mu)
    print(r2)
    r2 = r2_score(objective.ydata, in_sample_mu.ravel())
    print(r2)
    r2 = r2_score(objective.ydata, np.mean(objective.ydata) * np.ones_like(objective.ydata))
    print(r2)
    # print(np.abs(objective.ydata - loo_mu))


if __name__ == '__main__':
    main()
