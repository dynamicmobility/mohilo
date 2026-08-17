"""Two trials end to end, against a fake device and two fake instruments.
"""

import time
import numpy as np
from functools import partial
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler
import torch
import pypolar as plr

DIM             = 3
BOX             = 5.0
SEED            = 95
GP_NOISE        = plr.NoiseModel.pinned(0.5)
MIN_LENGTHSCALE = 0.2
NUM_QUERIES     = 15
ACQ_STRAT       = 'lognei'

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
        input('Send action?')
        print('EXO GOT ACTION', action)


def get_data_from_cart(action: np.ndarray):
    return METABOLIC_TRUTH(action)


def ask_subject(action: np.ndarray):
    return COMFORT_TRUTH(action)

def make_probes():
    probes = [
        plr.Probe(name='Metabolic Cart', caller=get_data_from_cart, obj_name='cost'),
        # plr.Probe(name='Survey', caller=ask_subject, repeats=4, obj_name='comfort'),
    ]
    return probes


def make_experiment(probes: list[plr.Probe]):
    experiment = plr.Logger(
        objectives   = [
            # pinned so the normalized frame, and so MIN_LENGTHSCALE, holds
            # still as measurements arrive
            plr.Objective.from_empty(name='cost', maximize=False,
                                     action_bounds=(-BOX, BOX)),
            # plr.Objective.from_empty(name='comfort', maximize=True,
            #                          action_bounds=(-BOX, BOX))
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
        'ucb'    : partial(UpperConfidenceBound, beta=UCB_BETA),
        'logei'  : LogExpectedImprovement,
        'lognei' : partial(LogNoisyExpectedImprovement, num_fantasies=NUM_FANTASIES),
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
        if i == 0:
            # Random action for the first
            action = plr.sample_actions(bounds=np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float), n=1, kind='uniform', seed=SEED)[0]
        else:
            # fit gp + Acquisition strategy for the rest
            gp = fit_gp(experiment.objectives['cost'])
            action = acqf.query(gp, q=1)[0]
        
        experiment.begin_trial(
            action=action,
            args={
                'Metabolic Cart': (action,),
                'Survey'        : (action,)
            }
        )
        
        start = time.time()
        while not experiment.all_measurements_completed:
            print('Waiting...', round(time.time() - start, 1))
            time.sleep(0.1)

        experiment.end_trial() # updates the objectives
    
    return experiment

def main():
    torch.manual_seed(SEED)     # optimize_acqf seeds its restarts from this

    probes = make_probes()
    experiment = make_experiment(probes)
    acqf = plr.AcquisitionFunction(
        acqf      = acquisition_factory(strategy=ACQ_STRAT, seed=SEED),
        objective = experiment.objectives['cost']
    )

    experiment = run_experiment(experiment, acqf)

    for name in experiment.objectives.names:
        objective = experiment.objectives[name]
        print(name, 'values ', objective.ydata)
        print(name, 'actions', objective.xdata.tolist())

    print('done')


if __name__ == '__main__':
    main()
