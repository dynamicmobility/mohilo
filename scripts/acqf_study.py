import time
import warnings
from dataclasses import asdict
from inspect import signature
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


def acquisition_spec(strategy, seed, **kwargs):
    """Every argument `acquisition_factory_1d` is called with, its defaults
    resolved, so a saved run rebuilds the acquisition it actually queried."""
    bound = signature(plr.acquisition_factory_1d).bind(strategy=strategy, seed=seed,
                                                       **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)

def fit_gp(objective, noise=None, hypers=None):
    return plr.BoTorchGP(
        objective           = objective,
        noise               = GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = MIN_LENGTHSCALE,
    )

def recommendation(gp, ground_truth):
    """The action the GP recommends now, and its regret in groundtruth spreads."""
    recommended, _, _ = gp.recommend(raw=True)

    return recommended[0], plr.normalized_inference_regret(
        raw_recommended_action   = recommended[0],
        ground_truth             = ground_truth
    )


def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    ground_truths     : dict[str, plr.SyntheticOracle]
):
    gp = None
    for i in tqdm(range(NUM_QUERIES)):
        objective = experiment.objectives[hilo.COMFORT]
        recommended, regret = None, None
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
            source = dataset.acquisition['strategy']
            gp = fit_gp(objective)
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action=action,
            args={
                hilo.METABOLIC: (action,),
                hilo.COMFORT:   (action, i + 1)
            }
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

        gp = fit_gp(experiment.objectives[hilo.COMFORT])
        recommended, regret = recommendation(gp, ground_truths[hilo.COMFORT])
        # the state the action was chosen from, recorded before it is applied
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
            recommended = recommended,
            regret      = regret
        )

    # the run's final state: every measurement, and the fit to all of them
    gp = fit_gp(experiment.objectives[hilo.COMFORT])
    recommended, regret = recommendation(gp, ground_truths[hilo.COMFORT])
    dataset.add_trial(
        objectives  = experiment.objectives,
        gp          = gp,
        recommended = recommended,
        regret      = regret
    )

    return experiment, gp, dataset

def setup_experiment(acq_strat, seed):
    probes     = hilo.make_probes()
    experiment = hilo.make_experiment(probes)
    acqf       = plr.AcquisitionFunction(
        acqf = plr.acquisition_factory_1d(strategy=acq_strat, seed=seed)
    )

    return experiment, acqf

def run_trial(acq_strat, trial):
    print(f'Trying {acq_strat} on trial {trial}')
    experiment, acqf = setup_experiment(
        acq_strat   = acq_strat,
        seed        = hilo.SEED
    )

    dataset = plr.ExperimentDataset(
        name         = acq_strat,
        acquisition  = acquisition_spec(
            strategy    = acq_strat,
            seed        = hilo.SEED,
            **ACQ_KWARGS
        ),
        groundtruths = hilo.GROUND_TRUTH_SPECS,
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
        acqf              = acqf,
        dataset           = dataset,
        ground_truths     = hilo.GROUND_TRUTHS
    )
    # mu, std, models = plr.loo(gp.objective, fit_gp, noise=GP_NOISE)
    # resid = mu - gp.objective.ydata
    # print('SCORE', r2_score(gp.objective.ydata, mu))
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
