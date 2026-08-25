import time
import warnings
from dataclasses import asdict, dataclass
from inspect import signature
from pathlib import Path
from sklearn.metrics import r2_score

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
TRUE_NOISE       = 0.25
NUM_QUERIES      = DIM * 13
ACQ_STRATS       = ['ucb', 'logei', 'qlognei', 'ts']
REPEATS          = 1
COMFORT          = 'Comfort'
METABOLIC        = 'Cost'
MULTITHREAD      = False

RUNS_PER_ACQF = 5

OUTPUT_DIR       = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS       = {}   # acquisition knobs overriding acquisition_factory_1d's own

GROUND_TRUTH_SPECS = { # TODO: change this so that the experimentdataset just stores a discretization version of this (points vs reconstructing the whole function)
    METABOLIC : {
        'func': 'Levy', 
        'dim': DIM, 
        'box': BOX, 
        'seed': SEED,
        'rel_noise_std': TRUE_NOISE
    },
    COMFORT   : {
        'func': 'Levy', 
        'dim': DIM, 
        'box': BOX, 
        'seed': SEED,
        'rel_noise_std': TRUE_NOISE
    },
}

GROUND_TRUTHS   = {name: plr.SyntheticOracle.from_name(**spec)
                   for name, spec in GROUND_TRUTH_SPECS.items()}
METABOLIC_TRUTH = GROUND_TRUTHS[METABOLIC]
COMFORT_TRUTH   = GROUND_TRUTHS[COMFORT]

@dataclass
class Simulation1D:
    gp_noise          : plr.NoiseModel
    min_lengthscale   : float
    num_queries       : int
    repeats           : int
    dim               : int = DIM
    seed              : int = SEED
    true_noise        : float = TRUE_NOISE
    noise_measure     : str = 'range'




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