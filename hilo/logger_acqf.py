import time
from functools import partial
from pathlib import Path

import numpy as np
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler
import torch
from pathlib import Path
import pypolar as plr

DIM              = 3
BOX              = 5.0
SEED             = 95
GP_NOISE         = plr.NoiseModel.prior(0.3)
MIN_LENGTHSCALE  = 0.3
NUM_QUERIES      = 30
ACQ_STRAT        = 'lognei'
REPEATS          = 1
SURVEY_TIMEOUT   = 5.0 
SURVEY_PERIOD    = 10.0
METABOLIC_PERIOD = 20.0
COMFORT          = 'Comfort'
METABOLIC        = 'Cost'

# Acquisition function stuff
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

METABOLIC_TRUTH = plr.SyntheticFunction(
    truth = plr.construct_function(
        func = plr.SYNTHETIC_1D_FUNCTIONS['Levy'],
        dim  = DIM,
        box  = BOX,
        seed = SEED
    ),
    rel_noise_std = 0.5,
)

COMFORT_TRUTH = plr.SyntheticFunction(
    truth = plr.construct_function(
        func = plr.SYNTHETIC_1D_FUNCTIONS['Levy'],
        dim  = DIM,
        box  = BOX,
        seed = SEED
    ),
    rel_noise_std = 0.3,
)

def make_probes():
    probes = [
        plr.Probe(
            name     = METABOLIC,
            caller   = METABOLIC_TRUTH,
            obj_name = METABOLIC,
        ),
        plr.Probe(
            name     = COMFORT,
            caller   = COMFORT_TRUTH,
            repeats  = REPEATS,
            obj_name = COMFORT
        ),
    ]
    return probes


def make_experiment(probes: list[plr.Probe]):
    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(
                name          = METABOLIC,
                maximize      = False,
                action_bounds = (-BOX, BOX)
            ),
            plr.Objective.from_empty(
                name          = COMFORT,
                maximize      = False,
                action_bounds = (-BOX, BOX)
            )
        ],
        probes       = probes,
        device       = plr.Device(),
        action_names = [f'x{i}' for i in range(DIM)]
    )
    return experiment

def acquisition_factory(strategy, seed):
    if strategy == 'lognei' and plr.NoiseModel.coerce(GP_NOISE).is_fitted:
        raise ValueError("'lognei' needs a FixedNoiseGaussianLikelihood, so set "
                         'GP_NOISE = plr.NoiseModel.pinned(...) to use it')

    factories = {
        'ucb'    : partial(
            UpperConfidenceBound, 
            beta = UCB_BETA
        ),
        'logei'  : LogExpectedImprovement,
        'lognei' : partial(
            LogNoisyExpectedImprovement, 
            num_fantasies = NUM_FANTASIES
        ),
        'qlognei': partial(
            qLogNoisyExpectedImprovement,
            sampler        = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=seed),
            prune_baseline = PRUNE_BASELINE
        )
    }
    if strategy not in factories:
        raise ValueError(f'no acquisition for {strategy!r}')

    return factories[strategy]

def fit_gp(objective):
    return plr.BoTorchGP(
        objective           = objective,
        noise               = GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = MIN_LENGTHSCALE,
        # signal_var=1.0,
        # length_scale=0.5
    )

def run_experiment(experiment: plr.Logger, acqf: plr.AcquisitionFunction):
    for i in range(NUM_QUERIES):
        objective = experiment.objectives[COMFORT]
        if not len(objective.ydata):
            # randomly sample if no data is collected
            action = plr.sample_actions(
                bounds = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float),
                n      = 1,
                kind   = 'uniform',
                seed   = SEED + i
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            gp = fit_gp(objective)
            action = acqf.query(gp, q=1)[0]

        print(action)
        experiment.begin_trial(
            action=action,
            args={
                METABOLIC: (action,),
                COMFORT:   (action, i + 1)
            }
        )
        
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives
        gp = fit_gp(objective)
    
    return experiment, gp

def make_fit_figure(truth, objective, gp, inferred, box, output):
    """The 1D fit: the GP and the truth evaluated on a grid, drawn by
    `plr.plot_fit_1d`.
    """
    output = Path(output)
    grid    = np.linspace(box[0, 0], box[1, 0], 1024)[:, None]
    mu, std = gp.posterior_at(grid, raw=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plr.plot_fit_1d(
        ax       = ax,
        x        = grid[:, 0],
        mu       = mu[:, 0],
        std      = std[:, 0],
        xdata    = objective.xdata[:, 0],
        ydata    = objective.ydata,
        truth    = plr.truth_at(truth, grid),
        # one objective, so the single column is the path itself
        paths    = gp.sample_paths(grid, 16, raw=True)[:, :, 0],
        vlines   = {'recommended action': inferred[0],
                    'true optimizer'    : truth.optimizers[0, 0].item()},
        band_std = 1.0,
        title    = f'{'f'} on Levy, 1D, after {len(objective.ydata)} measurements'
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def main():
    torch.manual_seed(SEED)

    probes     = make_probes()
    experiment = make_experiment(probes)
    acqf       = plr.AcquisitionFunction(
        acqf      = acquisition_factory(strategy='qlognei', seed=SEED),
        objective = experiment.objectives[COMFORT],
        raw_samples=2048,
        num_restarts=32
    )

    experiment, gp = run_experiment(experiment, acqf)
    best, mu, std = gp.best_actions(raw=True)
    # (2, DIM) of [lower; upper] rows
    box = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float)

    if DIM == 1:
        make_fit_figure(COMFORT_TRUTH.truth, experiment.objectives[COMFORT], gp, inferred=best, box=box, output='here.svg')
    print('here', best)
    print('here2', COMFORT_TRUTH(best, noise=False))
    actions = plr.sample_actions(
        bounds = [[-BOX]*DIM, [BOX]*DIM],
        n=1024,
        kind='sobol',
        seed=SEED
    )
    values = COMFORT_TRUTH([actions], noise=False)
    print(values.max() - values.min())


if __name__ == '__main__':
    main()
