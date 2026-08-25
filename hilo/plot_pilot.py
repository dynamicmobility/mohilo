from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from sklearn.metrics import r2_score

import pypolar as plr
from pypolar.optimization.gp import DTYPE

# TODO: this needs to be remade with up-to-date code and plot metrics + LOO + recommendations etc...
# you should create an example dataset using synthetics for testing.

ACTIONS         = ['h_flex_torque_scale', 'h_ext_torque_scale', 'hip_delay_idx']
OBJS_PATH       = Path('human_data/pilot_mohilo.csv')
ACTIONS_PATH    = Path('human_data/MH01_walk.csv')

CONTOUR_RES     = 200                       # plotting only
SEED            = 95

MIN_LENGTH_SCALE = 0.1                          # lengthscale floor, normalized action frame
GP_NOISE         = plr.NoiseModel.prior(0.3)    # fitted under a prior on 30% of each objective's spread


def compute_hypervolume(Y, ref=None):
    """Finds the hypervolume dominated by Y in reference to ref"""
    Y = torch.as_tensor(np.atleast_2d(Y), dtype=DTYPE)
    ref = ref if ref is not None else torch.zeros((Y.shape[1],))
    return DominatedPartitioning(ref_point=ref, Y=Y).compute_hypervolume().item()

def posterior_mu_at(model: ModelListGP, X, chunk=2048):
    """Per-objective posterior mean at X. Vectorized evaluation in sizes of chunk"""
    X = torch.as_tensor(np.atleast_2d(X), dtype=DTYPE).unsqueeze(1)
    with torch.no_grad():
        out = [model.posterior(X[i:i+chunk]).mean.squeeze(1) for i in range(0, X.shape[0], chunk)]
    
    return torch.cat(out).numpy()

def read_data(objs_path: Path, actions_path: Path) -> plr.DecoupledObjectives:
    objs_df = pd.read_csv(objs_path)
    actions_df = pd.read_csv(actions_path)
    
    objectives = plr.DecoupledObjectives.from_empty()
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Cost'],
        maximize  = False,
        name      = 'Metabolic Cost'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Speed'],
        maximize  = False,
        name      = '10m walk test'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Comfort Treadmill'],
        maximize  = True,
        name      = 'Comfort Treadmill'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Comfort Floor'],
        maximize  = True,
        name      = 'Comfort Floor'
    ))
    
    return objectives

def fit_gp(
    objective   : plr.Objective,
    noise       : plr.NoiseModel,
    hypers      : plr.GPHyperparameters = None
):
    """A GP on `objective`, with its hyperparameters refit or frozen.

    Args:
        objective: the measurements to condition on.
        noise: a `NoiseModel`, or a fraction of the objective's spread to pin,
            or None to fit it by marginal likelihood alongside the kernel.
        hypers: a `GPHyperparameters` to hold fixed, or None to fit them. Every
            field supersedes the arguments, the noise included, so a fitted
            noise freezes across folds exactly as the kernel does.
    """
    if hypers is None:
        return plr.BoTorchGP(objective, noise=noise, fit_hyperparameters=True,
                             min_length_scale=MIN_LENGTH_SCALE)

    # Standardize divides train_Y by its own sample spread, so the square root
    # of a post-Standardize noise variance is the fraction a pinned noise states
    return plr.BoTorchGP(
        objective,
        noise               = plr.NoiseModel.pinned(np.sqrt(hypers.noise_var)),
        fit_hyperparameters = False,
        length_scale        = hypers.lengthscale,
        signal_var          = hypers.signal_var,
        min_length_scale    = MIN_LENGTH_SCALE
    )
    

def report_loo(objectives: plr.DecoupledObjectives, noise: plr.NoiseModel):
    """Leave-one-out accuracy and calibration, one row per objective.

    Every fold refits through `fit_gp`, which is the configuration `main` fits,
    so the numbers score the model in use rather than a different one.
    """
    print('\n=== leave-one-out ===')
    print(f'  {"objective":<20}{"R2":>8}{"RMSE":>10}{"sd(y)":>10}{"z-std":>8}')
    for i in range(len(objectives)):
        obj = objectives[i]
        mu, std, models = plr.loo(obj, fit_gp, noise=noise)

        resid = mu - obj.ydata

        # loo reports the latent std, but a residual carries the observation
        # noise too, so the z divides by the predictive std. noise_var is
        # post-Standardize, hence the scale back into the objective's units
        hypers    = [gp.get_fitted_hyperparameters() for gp in models]
        noise_std = obj.ytransform.inv_scale(np.array(
            [np.sqrt(h.noise_var) * h.standardize_scale for h in hypers]))
        z = resid / np.hypot(std, noise_std)

        print(f'  {obj.name:<20}{r2_score(obj.ydata, mu):>+8.3f}'
              f'{np.sqrt(np.mean(resid ** 2)):>10.3f}'
              f'{obj.ydata.std():>10.3f}{z.std():>8.2f}')


def main():
    # read in the data
    objectives = read_data(
        actions_path    = ACTIONS_PATH,
        objs_path       = OBJS_PATH
    )
    
    OBJ1 = 'Metabolic Cost'
    OBJ2 = 'Comfort Treadmill'
    
    # OBJ1 = '10m walk test'
    # OBJ2 = 'Comfort Floor'
        
    mogp = plr.DecoupledMOGP(
        objectives=objectives[[OBJ1, OBJ2]],
        fit_hyperparameters=True,
        noise=GP_NOISE,
        min_length_scale=MIN_LENGTH_SCALE
    )

    report_loo(mogp.objectives, GP_NOISE)
    quit()

    nd_objs = objs[plr.get_nondominated(objs)]
    
    # plot them
    fig, ax = plt.subplots()
    ax.scatter(objectives[OBJ1].ytransform.inv(objs[:, 0]), objectives[OBJ2].ytransform.inv(objs[:, 1]), s=5, alpha=0.1)
    ax.scatter(objectives[OBJ1].ytransform.inv(nd_objs[:, 0]), objectives[OBJ2].ytransform.inv(nd_objs[:, 1]), s=25, c='red')
    fig.savefig('hilo/output/test.png', dpi=500)
    
    # print best of each
    actions, mu, std = mogp.recommend(raw=True)
    width = max(len(name) for name in mogp.objectives.names) + 6   # room for ' (min)'

    print('\n=== best action per objective ===')
    print(f'  {"objective":<{width}} ' + ''.join(f'{col:>22}' for col in ACTIONS)
          + f'{"predicted":>22}{"best measured":>16}')
    for i in range(len(mogp.objectives)):
        obj = mogp.objectives[i]
        label = f'{obj.name} ({"max" if obj.maximize else "min"})'
        best_seen = obj.ydata.max() if obj.maximize else obj.ydata.min()
        print(f'  {label:<{width}} ' + ''.join(f'{a:>22.4f}' for a in actions[i])
              + f'{mu[i]:>13.3f} +/- {std[i]:<5.3f}'
              + f'{best_seen:>16.3f}')


if __name__ == '__main__':
    main()