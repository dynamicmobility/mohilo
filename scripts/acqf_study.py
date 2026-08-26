import time
import warnings
from dataclasses import asdict
from pathlib import Path
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
from linear_operator.utils.warnings import NumericalWarning
import torch
import pypolar as plr
from tqdm import tqdm
import hilo.simulation as hilo
warnings.filterwarnings('ignore', category=NumericalWarning)

DIM              = 3
GP_NOISE         = plr.NoiseModel.prior(0.5)
MIN_LENGTHSCALE  = 0.1
NUM_QUERIES      = DIM * 13
ACQ_STRATS       = ['ucb', 'logei', 'qlognei', 'ts']

RUNS_PER_ACQF = 5
REPEATS          = 1
OUTPUT_DIR       = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS       = {}   # acquisition knobs overriding acquisition_factory_1d's own


def fit_gp(objective, noise=None, hypers=None):
    return plr.BoTorchGP(
        objective           = objective,
        noise               = GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = MIN_LENGTHSCALE,
    )

def aux(gp, ground_truth):
    """What a trial is scored with: the action the GP recommends now, and its
    regret in groundtruth spreads."""
    recommended, _, _ = gp.recommend(raw=True)

    return {
        'recommended' : recommended[0],
        'regret'      : plr.normalized_inference_regret(
            raw_recommended_action   = recommended[0],
            ground_truth             = ground_truth
        )
    }


def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    ground_truth      : plr.SyntheticOracle
):
    gp = None
    for i in tqdm(range(NUM_QUERIES)):
        objective = experiment.objectives[hilo.COMFORT]
        if len(objective.ydata) < 1:
            # randomly sample if no data is collected
            source = 'random'
            action = plr.sample_actions(
                dim    = DIM,
                n      = 1,
                kind   = 'uniform',
                seed   = hilo.SEED + i,
                bounds = experiment.objectives.action_bounds
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition.strategy
            gp = fit_gp(objective)
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action=action,
            args={hilo.COMFORT: (action, i + 1)}
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

        gp = fit_gp(experiment.objectives[hilo.COMFORT])
        # the state the action was chosen from, recorded before it is applied
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
            aux         = aux(gp, ground_truth)
        )

    # the run's final state: every measurement, and the fit to all of them
    gp = fit_gp(experiment.objectives[hilo.COMFORT])
    dataset.add_trial(
        objectives  = experiment.objectives,
        gp          = gp,
        aux         = aux(gp, ground_truth)
    )

    return experiment, gp, dataset

def setup_experiment(acq_strat, seed):
    probes      = hilo.make_probes()
    experiment  = hilo.make_experiment(probes)
    acqf_params = plr.AcquisitionParams(
        strategy = acq_strat,
        seed     = seed,
        **ACQ_KWARGS
    )

    return experiment, acqf_params

def run_trial(acq_strat, trial):
    print(f'Trying {acq_strat} on trial {trial}')
    experiment, acqf_params = setup_experiment(
        acq_strat   = acq_strat,
        seed        = hilo.SEED
    )

    dataset = plr.ExperimentDataset(
        name         = acq_strat,
        acquisition  = acqf_params,
        groundtruth  = hilo.GROUND_TRUTH_1D,
        config       = asdict(
            hilo.Simulation1D(
                gp_noise          = plr.NoiseModel.coerce(GP_NOISE),
                min_lengthscale   = MIN_LENGTHSCALE,
                num_queries       = NUM_QUERIES,
                repeats           = REPEATS,
            )
        ),
        path         = OUTPUT_DIR / f'{acq_strat}-{trial}.json'
    )

    experiment, gp, dataset = run_experiment(
        experiment        = experiment,
        acqf              = acqf_params.build(),
        dataset           = dataset,
        ground_truth      = hilo.COMFORT_TRUTH
    )
    print(f'wrote {dataset.save()}')

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(hilo.SEED)
    
    for acq_strat in ACQ_STRATS:
        for trial in range(RUNS_PER_ACQF):
            run_trial(acq_strat, trial)

    print('Done')


if __name__ == '__main__':
    main()
