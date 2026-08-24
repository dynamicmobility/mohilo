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
import pypolar as plr
from tablet.survey import Survey

# TODO: rename and setup for the actual hardware run. make an emulation mode

DIM              = 3
BOX              = 5.0
SEED             = 95
GP_NOISE         = plr.NoiseModel.pinned(0.5)
MIN_LENGTHSCALE  = 0.2
NUM_QUERIES      = 15
ACQ_STRAT        = 'lognei'
REPEATS          = 1
SURVEY_TIMEOUT   = 10.0 
SURVEY_PERIOD    = 15.0
METABOLIC_PERIOD = 15.0
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
        func = plr.SYNTHETIC_1D_FUNCTIONS['DixonPrice'],
        dim  = DIM,
        box  = BOX,
        seed = SEED
    ),
    rel_noise_std = 0.3,
)

class Exo(plr.Device):

    def send(self, action):
        print('Exo got action', action)
        input('Send action? ')


def get_data_from_cart(action: np.ndarray):
    time.sleep(METABOLIC_PERIOD)
    return METABOLIC_TRUTH(action)

def make_probes(survey: Survey):
    probes = [
        plr.Probe(
            name     = METABOLIC,
            caller   = get_data_from_cart,
            obj_name = METABOLIC,
        ),
        plr.Probe(
            name     = COMFORT,
            caller   = survey.ask,
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
                maximize      = True,
                action_bounds = (-BOX, BOX)
            )
        ],
        probes       = probes,
        device       = Exo(),
        action_names = ['x', 'y', 'z']
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

        experiment.begin_trial(
            action=action,
            args={
                METABOLIC: (action,),
                COMFORT:   (action, i + 1)
            }
        )
        
        start = time.time()
        while not experiment.all_measurements_completed:
            print('Waiting...', round(time.time() - start, 1), end='\t\t\r')
            time.sleep(0.1)

        experiment.end_trial() # updates the objectives
    
    return experiment

def connect_to_ipad():
    ipad = Survey(timeout=SURVEY_TIMEOUT, period=SURVEY_PERIOD)
    print('Waiting for the iPad...')
    ipad.wait_for_ipad()
    return ipad

def main():
    torch.manual_seed(SEED)

    ipad       = connect_to_ipad()
    probes     = make_probes(ipad)
    
    experiment = make_experiment(probes)
    acqf       = plr.AcquisitionFunction(
        acqf      = acquisition_factory(strategy=ACQ_STRAT, seed=SEED),
        objective = experiment.objectives[COMFORT]
    )

    try:
        experiment = run_experiment(experiment, acqf)
    finally:
        ipad.close()

    for name in experiment.objectives.names:
        objective = experiment.objectives[name]
        print(name, 'values ', objective.ydata)
        print(name, 'actions', objective.xdata.tolist())


if __name__ == '__main__':
    main()
