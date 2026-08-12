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
NOISE        = 0.5     # noise added, as a fraction of the truth's spread
GP_NOISE     = 0.1     # noise the GP assumes; None matches NOISE
REFIT_FOLDS  = False    # refit kernel hyperparameters inside every LOO fold
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
DESIGN       = 'sobol'  # 'sobol' (space-filling) or 'uniform' (iid)
SEED         = 95
OUTPUT       = Path('hilo/output/gp_diagnostics.png')


LENGTHSCALE = 0.2
SIGNAL_VAR = 1.0
HYPERS = {
    'lengthscale': LENGTHSCALE,
    'signal_var': SIGNAL_VAR
}
HYPERS = None


PLOT_RES     = 400      # plotting only
SPREAD_REF   = 4096     # samples used to measure the truth's spread

# Okabe-Ito, a published colorblind-safe palette
TRUTH_COLOR = '#000000'
FOLD_COLOR  = '#56B4E9'
FULL_COLOR  = '#D55E00'
POINT_COLOR = '#0072B2'


def fit_gp(objective, noise_std, hypers=None):
    """A GP on `objective`, with kernel hyperparameters refit or frozen.

    Args:
        objective: the measurements to condition on.
        noise_std: observation noise, as a fraction of the objective's spread.
        hypers: a `get_fitted_hyperparameters()` dict to hold fixed, or None to
            fit them.
    """
    gp = plr.BoTorchGP(objective, noise_std=noise_std, fit_hyperparameters=hypers is None)
    if hypers is not None:
        gp.LENGTH_SCALE, gp.SIGNAL_VAR = hypers['lengthscale'], hypers['signal_var']
        gp.update_feedback(objective)

    return gp


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
        objective = objectives['Metabolic Cost']
        objective = read_MT0x_data(
            # Path('human_data/MT03_incline.csv'),
            # Path('human_data/MT04_incline.csv'),
            Path('human_data/MT05_incline.csv'),
            num_samples=NUM_SAMPLES,
            seed=94
        )

    full_gp = fit_gp(objective, GP_NOISE, hypers=HYPERS)
    hypers  = HYPERS if REFIT_FOLDS else full_gp.get_fitted_hyperparameters()

    loo_mu, loo_std, models = leave_one_out(objective, GP_NOISE, hypers)

    r2 = r2_score(objective.ydata, loo_mu)
    print(r2)


if __name__ == '__main__':
    main()
