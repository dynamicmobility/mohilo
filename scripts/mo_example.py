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
import hilo.shared.simulation as hilo
warnings.filterwarnings('ignore', category=NumericalWarning)

# TODO: go through all the reference setting/computing and min/max objective logic in this codebase
# ACQ_STRATS    = ['qlognehvi', 'qlognparego', 'qhvkg', 'qlogehvi']
ACQ_STRATS    = ['qlognparego']
RUNS_PER_ACQF = 1
OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS    = {}   # acquisition knobs overriding acquisition_factory_1d's own

def aux(
    gp              : plr.DecoupledMOGP,
    ground_truth    : plr.MOSyntheticOracle
):
    queried = np.vstack([o.xdata for o in gp.objectives.objectives])

    mu, _       = gp.posterior_at(ground_truth.scan_actions, raw=True)
    front       = plr.get_nondominated(gp.objectives.maximization_space(mu))
    recommended = ground_truth.scan_actions[front]

    return {
        'hv_regret'          : plr.normalized_hypervolume_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = gp.objectives,
            ref_point       = None #hilo.REF_POINT
        ),
        'hv_regret_attained' : plr.attained_hypervolume_regret(
            raw_actions     = queried,
            ground_truth    = ground_truth,
            objectives      = gp.objectives,
            ref_point       = None #hilo.REF_POINT
        ),
        'front_alignment'    : plr.front_alignment_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = gp.objectives
        ),
        # precision and coverage read opposite ways: a tight front scores well
        # on alignment and badly here, a broad one with strays the other way
        'front_coverage'     : plr.front_coverage_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = gp.objectives
        )
    }


def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    ground_truth      : plr.MOSyntheticOracle
):
    gp = None
    for i in tqdm(range(hilo.NUM_QUERIES)):
        if i < hilo.NUM_RANDOM:
            # randomly sample until the acquisition has something to fit to
            source = 'random'
            action = plr.sample_actions(
                dim    = hilo.DIM,
                n      = 1,
                kind   = 'uniform',
                seed   = hilo.SEED + i,
                bounds = experiment.objectives.action_bounds
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition.strategy
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action = action,
            args           = {
                hilo.METABOLIC: (action, i + 1),
                hilo.COMFORT:   (action, i + 1, hilo.SURVEY_TIMEOUT, hilo.SURVEY_PERIOD)
            }
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

        gp = plr.DecoupledMOGP(
            objectives          = experiment.objectives,
            noise               = hilo.GP_NOISE,
            fit_hyperparameters = True,
            min_length_scale    = hilo.MIN_LENGTHSCALE,
        )
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
            aux         = aux(gp, ground_truth)
        )

    return experiment, gp, dataset

def setup_experiment(acq_strat, seed):
    probes     = hilo.make_probes_mo()
    experiment = hilo.make_experiment_mo(
        probes    = probes,
        maximize  = hilo.MAXIMIZE
    )
    params     = plr.AcquisitionParams(
        strategy          = acq_strat,
        seed              = seed,
        num_objectives    = 2,
        raw_ref_point     = None #hilo.REF_POINT,
        **ACQ_KWARGS
    )
    assert experiment.objectives.names == list(hilo.GROUND_TRUTH_PARAMS.objectives)

    return experiment, params

def run_trial(acq_strat, trial):
    print(f'Trying {acq_strat} on trial {trial}')
    experiment, params = setup_experiment(
        acq_strat   = acq_strat,
        seed        = hilo.SEED + trial
    )

    dataset = plr.ExperimentDataset(
        name         = acq_strat,
        acquisition  = params,
        groundtruth  = hilo.GROUND_TRUTH_PARAMS,
        config       = asdict(
            hilo.Config(
                gp_noise          = plr.NoiseModel.coerce(hilo.GP_NOISE),
                min_lengthscale   = hilo.MIN_LENGTHSCALE,
                num_queries       = hilo.NUM_QUERIES,
                repeats           = hilo.REPEATS,
                dim               = hilo.DIM,
                seed              = hilo.SEED
            )
        ),
        path         = OUTPUT_DIR / f'{acq_strat}-{trial}.json'
    )

    experiment, gp, dataset = run_experiment(
        experiment        = experiment,
        acqf              = params.build(),
        dataset           = dataset,
        ground_truth      = hilo.MO_TRUTH
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
