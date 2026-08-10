from pathlib import Path
import pandas as pd
import numpy as np
from config.hipexo import hipexo_pilot
import pypolar as plr
from pypolar.optimization.objectives import (
    AffineTransform,
    Objective,
    DecoupledObjectives
)
from hilo.create import create_pilot_regression
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc



from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.multi_objective import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean

ACTIONS         = ['h_flex_torque_scale', 'h_ext_torque_scale', 'hip_delay_idx']
OBJS_PATH       = Path('human_data/pilot_mohilo.csv')
ACTIONS_PATH    = Path('human_data/MH01_walk.csv')

NOISE_STD       = 0.05
RAW_SAMPLES     = 512                       # stage-1 Sobol samples per query
NUM_RESTARTS    = 8                         # stage-2 L-BFGS-B starting points
XI              = 0.01                      # EI improvement threshold
CONTOUR_RES     = 200                       # plotting only
SEED            = 95
DTYPE           = torch.float64

def build_so_model(
    X             : np.ndarray | torch.Tensor,
    Y             : np.ndarray | torch.Tensor,
    sigma2        : float,
    signal_var    : float,
    length_scale  : float
): 
    """Builds a fixed hyperparameter exact (closed form) GP over the observations"""
    # Convert data into torch tensors
    train_X = torch.as_tensor(X, dtype=DTYPE)
    train_Y = torch.as_tensor(Y, dtype=DTYPE)

    # Setup the prior covariance matrix/kernel
    prior_covar = ScaleKernel(RBFKernel()).to(DTYPE) # why do we need a ScaleKernel here?
    prior_covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
    prior_covar.outputscale = torch.tensor(signal_var ** 2, dtype=DTYPE)

    model = SingleTaskGP(
        train_X             = train_X,
        train_Y             = train_Y,
        train_Yvar          = torch.full_like(train_Y, sigma2),
        covar_module        = prior_covar,
        mean_model          = ZeroMean(),
        outcome_transform   = None,
        input_transform     = None
    )
    return model


def build_mo_model(
    X             : np.ndarray,
    Y             : np.ndarray,
    sigma2s       : np.ndarray,
    signal_vars   : np.ndarray,
    length_scales : np.ndarray
):
    train_X = torch.as_tensor(X, dtype=DTYPE)
    train_Y = torch.as_tensor(Y, dtype=DTYPE)

    models = []
    for j, (sv, ls, sg2) in enumerate(zip(signal_vars, length_scales, sigma2s)):
        models.append(build_so_model(
            X             = train_X,
            Y             = train_Y[:, j].reshape(-1, 1),
            sigma2        = sg2,
            signal_var    = sv,
            length_scale  = ls
        ))
    
    return ModelListGP(*models).eval()


def maximize(acqf, bounds):
    """Maximize an acquisition function over a hypercube"""
    candidate, val = optimize_acqf(
        acq_function    = acqf,
        bounds          = bounds,
        q               = 1,     # one action per query, not batched
        num_restarts    = NUM_RESTARTS,
        raw_samples     = RAW_SAMPLES,
    )
    return candidate.squeeze(0).detach().numpy()

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

def pilot_config(objectives, values, noise_frac=0.15):
    """A copy of the pilot config with GP hyperparameters matched to the data's scale."""
    cfg = hipexo_pilot.model_copy(deep=True)   # each fit gets its own config
    objective_range = float(np.mean([v.max() - v.min() for v in values]))

    lengthscale, signal_var, precision = plr.derive_gp_hyperparams(
        domain_size    = float(np.max(cfg.problem.action_high
                                         - cfg.problem.action_low)),
        expected_range = objective_range,
        noise_var      = objective_range * noise_frac
    )
    cfg.num_objs                   = len(objectives)
    cfg.problem.precisions         = np.full(cfg.num_objs, precision)
    cfg.optimizer.signal_variances = [signal_var] * cfg.num_objs
    cfg.optimizer.length_scales    = [lengthscale] * cfg.num_objs
    return cfg

def read_data(objs_path: Path, actions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Objectives]:
    objs_df = pd.read_csv(objs_path)
    actions_df = pd.read_csv(actions_path)
    
    objectives = Objectives.from_df(
        df = objs_df,
        columns = [
            'Cost', 
            'Speed', 
            'Comfort Treadmill', 
            'Comfort Floor'
        ],
        maximize = [False, False, True, True],
        names = [
            'Metabolic Cost', 
            '10m walk time', 
            'Comfort on Treadmill', 
            'Comfort on Floor'
        ]
    )
    
    return objs_df, actions_df, objectives


def get_mogp_hyperparameters(objectives: Objectives, actions_df: pd.DataFrame):
    action_set = actions_df[ACTIONS].to_numpy().min(axis=0)
    ls, sv, sg = [[]] * len(objectives)

    # domain_highs = actions_df[ACTIONS].to_numpy().min(axis=0),
    # domain_lows  = actions_df[ACTIONS].to_numpy().max(axis=0),
    
    for i in range(num_objs):
        lengthscale, signal_var, _ = plr.derive_gp_hyperparams(
            domain_size = objectives.d,
            expected_range=objectives.max() - objectives.min()
        )
    

def main():
    # read in the data
    objs_df, actions_df, objectives = read_data(
        actions_path    = ACTIONS_PATH,
        objs_path       = OBJS_PATH
    )
    
    print(actions_df[ACTIONS].to_numpy().min(axis=0),)
    quit()
    # create the MO GP
    mogp = create_mogp(
        domain_highs = actions_df[ACTIONS].to_numpy().min(axis=0),
        domain_lows  = actions_df[ACTIONS].to_numpy().max(axis=0),
        objective_highs = [metabolics.max(), comfort_treadmill.max()],
        objective_lows  = [metabolics.min(), comfort_treadmill.min()],
    )

    # fit the GPs
    mogp = fit_gp()

    # build the front

    # plot them

if __name__ == '__main__':
    main()