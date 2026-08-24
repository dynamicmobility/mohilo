import time
import warnings
from dataclasses import asdict
from functools import partial
from inspect import signature
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from linear_operator.utils.warnings import NumericalWarning
import torch
import pypolar as plr
from tqdm import tqdm
warnings.filterwarnings('ignore', category=NumericalWarning)

DIM              = 3
BOX              = 5.0
SEED             = 95
TRUE_NOISE       = 0.5
GP_NOISE         = plr.NoiseModel.prior(TRUE_NOISE)
# GP_NOISE         = plr.NoiseModel.pinned(TRUE_NOISE)
MIN_LENGTHSCALE  = 0.1
NUM_QUERIES      = DIM * 13
ACQ_STRATS       = ['ucb', 'logei', 'qlognei']
REPEATS          = 1
COMFORT          = 'Comfort'
METABOLIC        = 'Cost'
MULTITHREAD      = False

RUNS_PER_ACQF = 10

OUTPUT_DIR       = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS       = {}   # acquisition knobs overriding acquisition_factory_1d's own

# the arguments `plr.make_synthetic` builds each groundtruth from, rather than
# the instance alone, so a saved run carries what rebuilds it
GROUND_TRUTH_SPECS = {
    METABOLIC : {'func': 'Levy', 'dim': DIM, 'box': BOX, 'seed': SEED,
                 'rel_noise_std': TRUE_NOISE},
    COMFORT   : {'func': 'Levy', 'dim': DIM, 'box': BOX, 'seed': SEED,
                 'rel_noise_std': TRUE_NOISE},
}


GROUND_TRUTHS   = {name: plr.make_synthetic(**spec)
                   for name, spec in GROUND_TRUTH_SPECS.items()}
METABOLIC_TRUTH = GROUND_TRUTHS[METABOLIC]
COMFORT_TRUTH   = GROUND_TRUTHS[COMFORT]


def run_config():
    """The constants a run was made under."""
    return {
        'dim'             : DIM,
        'box'             : BOX,
        'seed'            : SEED,
        'true_noise'      : TRUE_NOISE,
        'gp_noise'        : asdict(plr.NoiseModel.coerce(GP_NOISE)),
        'min_lengthscale' : MIN_LENGTHSCALE,
        'num_queries'     : NUM_QUERIES,
        'repeats'         : REPEATS,
    }


def acquisition_spec(strategy, seed, **kwargs):
    """Every argument `acquisition_factory_1d` is called with, its defaults
    resolved, so a saved run rebuilds the acquisition it actually queried."""
    bound = signature(plr.acquisition_factory_1d).bind(strategy=strategy, seed=seed,
                                                       **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)

def make_probes():
    probes = [
        plr.Probe(
            name              = METABOLIC,
            caller            = METABOLIC_TRUTH,
            obj_name          = METABOLIC,
            separate_thread   = MULTITHREAD
        ),
        plr.Probe(
            name              = COMFORT,
            caller            = COMFORT_TRUTH,
            repeats           = REPEATS,
            obj_name          = COMFORT,
            separate_thread   = MULTITHREAD
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
        action_names = [f'x{i}' for i in range(DIM)],
    )
    return experiment

def fit_gp(objective):
    return plr.BoTorchGP(
        objective           = objective,
        noise               = GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = MIN_LENGTHSCALE,
        # signal_var=1.0,
        # length_scale=0.5
    )

def inference_regret(gp, optimal_actions, ground_truths, gt_spread):
    """The action the GP recommends now, and its groundtruth gap in spreads."""
    recommended, _, _ = gp.recommend(raw=True)
    comfort_gt        = ground_truths[COMFORT]
    optimal_val       = comfort_gt(optimal_actions[COMFORT], noise=False)
    inferred_val      = comfort_gt(recommended, noise=False)
    scaled_regret     = (inferred_val - optimal_val) / gt_spread[COMFORT]

    return recommended[0], np.ravel(scaled_regret)[0]


def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    optimal_actions   : dict[str, np.ndarray],
    ground_truths     : dict[str, plr.SyntheticFunction],
    gt_spread         : dict[str, float]
):
    gp = None
    for i in tqdm(range(NUM_QUERIES)):
        objective = experiment.objectives[COMFORT]
        recommended, regret = None, None
        if len(objective.ydata) < 1:
            # randomly sample if no data is collected
            source = 'random'
            action = plr.sample_actions(
                bounds = BOX,
                dim    = DIM,
                n      = 1,
                kind   = 'uniform',
                seed   = SEED + i
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition['strategy']
            gp = fit_gp(objective)
            degenerate_gp = gp.get_fitted_hyperparameters().signal_var < 1e-2
            action = acqf.query(gp, q=1)[0]
            # if degenerate_gp:
            #     print(action, i)
            #     print('IT HAPPENED\n\n')

            recommended, regret = inference_regret(gp, optimal_actions,
                                                   ground_truths, gt_spread)

        # the state the action was chosen from, recorded before it is applied
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
            recommended = recommended,
            regret      = regret
        )

        experiment.begin_trial(
            action=action,
            args={
                METABOLIC: (action,),
                COMFORT:   (action, i + 1)
            }
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

    # the run's final state: every measurement, and the fit to all of them
    gp = fit_gp(experiment.objectives[COMFORT])
    recommended, regret = inference_regret(gp, optimal_actions, ground_truths, gt_spread)
    dataset.add_trial(
        objectives  = experiment.objectives,
        gp          = gp,
        recommended = recommended,
        regret      = regret
    )

    return experiment, gp, dataset

def setup_experiment(acq_strat, seed):
    probes     = make_probes()
    experiment = make_experiment(probes)
    acqf       = plr.AcquisitionFunction(
        acqf      = plr.acquisition_factory_1d(strategy=acq_strat, seed=seed),
        objective = experiment.objectives[COMFORT],
    )

    return experiment, acqf


def get_groundtruth_optimum(
    seed,
    num_samples=4096
):
    actions = plr.sample_actions(
        bounds    = BOX,
        dim       = DIM,
        n         = num_samples,
        kind      = 'sobol',
        seed      = seed,
    )
    gt_values = np.array([GROUND_TRUTHS[obj_name](actions, noise=False) for obj_name in GROUND_TRUTHS.keys()])
    idxs = np.argmin(gt_values, axis=1)
    spread = gt_values.max(axis=1) - gt_values.min(axis=1)
    return (
        {name: actions[i] for name, i in zip(GROUND_TRUTHS.keys(), idxs, strict=True)},
        {name: sp for name, sp in zip(GROUND_TRUTHS.keys(), spread, strict=True)}
    )

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    optimal_actions, gt_spread = get_groundtruth_optimum(
        seed = SEED,
        num_samples=8192
    )
    
    for acq_strat in ACQ_STRATS:
        for trial in range(RUNS_PER_ACQF):
            print(f'Trying {acq_strat} on trial {trial}')
            experiment, acqf = setup_experiment(
                acq_strat   = acq_strat,
                seed        = SEED
            )
            dataset = plr.ExperimentDataset(
                name         = acq_strat,
                acquisition  = acquisition_spec(strategy=acq_strat, seed=SEED, **ACQ_KWARGS),
                groundtruths = GROUND_TRUTH_SPECS,
                config       = run_config(),
                path         = OUTPUT_DIR / f'{acq_strat}-{trial}.json'
            )
            experiment, gp, dataset = run_experiment(
                experiment        = experiment,
                acqf              = acqf,
                dataset           = dataset,
                optimal_actions   = optimal_actions,
                ground_truths     = GROUND_TRUTHS,
                gt_spread         = gt_spread
            )
            print(f'wrote {dataset.save()}')


if __name__ == '__main__':
    main()
