import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
from tqdm import tqdm
import hilo.shared.simulation as hilo
import scripts.mo_example as mo
from scripts.icra.synthetic_1d_plot import make_figures
warnings.filterwarnings('ignore', category=NumericalWarning)

OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')


def make_idealpoint_groundtruth(seed, box, noise, dim, optima, weights=None, offset=None,
                                low=None, high=None):
    GROUND_TRUTH_PARAMS = plr.SyntheticOracleParams(
        func          = 'IdealPoint',
        objectives    = ('Metabolic Cost', 'Comfort'),
        optima        = optima,
        weights       = weights,
        offset        = offset,
        low           = low,
        high          = high,
        dim           = dim,
        box           = box,
        seed          = seed,
        rel_noise_std = noise,
        measure = 'std'
    )

    gt = GROUND_TRUTH_PARAMS.build()
    return gt, GROUND_TRUTH_PARAMS

def make_probes_mo(gt: plr.MOSyntheticOracle, ipad=None, multithread=False):
    """One probe per objective, each measuring its own column of the MO truth.
    """
    probes = [
        plr.Probe(
            name              = hilo.METABOLIC,
            caller            = lambda action, trial_num: gt.get_oracle(0)(action),
            repeats           = 1,
            obj_name          = hilo.METABOLIC,
            separate_thread   = multithread
        ),
        plr.Probe(
            name              = hilo.COMFORT,
            caller            = lambda action, trial, timeout, period: gt.get_oracle(1)(action),
            repeats           = 1,
            obj_name          = hilo.COMFORT,
            separate_thread   = multithread
        )
    ]
    return probes


def setup_experiment(acq_strat, seed, gt: plr.SyntheticOracle):
    probes     = make_probes_mo(gt)
    experiment = hilo.make_experiment_mo(
        probes    = probes,
        maximize  = hilo.MAXIMIZE
    )
    params     = plr.AcquisitionParams(
        strategy          = acq_strat,
        seed              = seed,
        num_objectives    = 2,
        raw_ref_point     = None,
        **hilo.ACQ_KWARGS
    )
    # assert experiment.objectives.names == list(gt.objectives)

    return experiment, params

def run_mobo_trial(seed, groundtruth: plr.SyntheticOracle, gt_params: plr.SyntheticOracleParams):
    experiment, params = setup_experiment(
        acq_strat   = 'qlognparego',
        seed        = seed,
        gt          = groundtruth
    )

    dataset = plr.ExperimentDataset(
        name         = 'qlognparego',
        acquisition  = params,
        groundtruth  = gt_params,
        config       = mo.asdict(
            hilo.Config(
                gp_noise          = plr.NoiseModel.coerce(hilo.GP_NOISE),
                min_lengthscale   = hilo.MIN_LENGTHSCALE,
                num_queries       = hilo.NUM_QUERIES,
                repeats           = hilo.REPEATS,
                dim               = hilo.DIM,
                seed              = hilo.SEED
            )
        ),
        path         = OUTPUT_DIR / f'{seed}.json'
    )

    experiment, gp, dataset = mo.run_experiment(
        experiment        = experiment,
        acqf              = params.build(),
        dataset           = dataset,
        ground_truth      = groundtruth
    )
    print(f'wrote {dataset.save()}')

    dataset: plr.ExperimentDataset = dataset
    names = ('front_coverage',)
    curves = {n: dataset.get_aux(n) for n in names}
    curves['evals'] = np.arange(len(dataset)) + 1
    return curves



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    hilo.DIM            = 1
    hilo.BOUNDS         = plr.as_bounds((-3, 3), hilo.DIM)
    hilo.ACTION_NAMES   = [f'x{i}' for i in range(hilo.DIM)]
    hilo.TRUE_NOISE     = 0.1
    hilo.GP_NOISE       = plr.NoiseModel.prior(hilo.TRUE_NOISE)
    hilo.NUM_QUERIES    = 7
    hilo.MAXIMIZE       = {hilo.METABOLIC: False, hilo.COMFORT: True}


    gt, params = make_idealpoint_groundtruth(
        seed    = 95,
        box     = (-3, 3),
        noise   = 0.05,
        dim     = hilo.DIM,
        optima  = (-1.0, 1.0),
        weights = (0.1, -0.4),
        offset  = None,
        low     = (-1.0, -1.0),
        high    = (1.0, 1.0)
    )

    curves = run_mobo_trial(
        seed        = hilo.SEED,
        groundtruth = gt,
        gt_params   = params
    )

    dataset = plr.ExperimentDataset.load(OUTPUT_DIR / f'{hilo.SEED}.json')
    for path in make_figures(dataset):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
