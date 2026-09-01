import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
from tablet.survey import Survey
import numpy as np
import logging
from hilo.shared.log import TO_BOTH, logged_input, setup_logger

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=NumericalWarning)

MULTITHREAD = True 

# Actions
DIM              = 3
BOUNDS           = [[0.0, 0.0, 0.0], [0.30, 0.30, 75.0]]
ACTION_NAMES     = [
    'h_flex_torque_scale',
    'h_ext_torque_scale',
    'hip_delay_idx'
]

# Optimization
SEED             = 95
NUM_QUERIES      = 12
ACQ_STRAT        = 'qlognparego'
ACQ_KWARGS       = {}
GP_NOISE         = plr.NoiseModel.prior(0.5)
MIN_LENGTHSCALE  = 0.1
NUM_RANDOM       = 3

# Objectives
NUM_OBJECTIVES   = 2
COMFORT          = 'Comfort'
METABOLIC        = 'Metabolic Cost'
REPEATS          = {METABOLIC: 1, COMFORT: 3}
MAXIMIZE         = {METABOLIC: False, COMFORT: True}
SURVEY_TIMEOUT   = 60.0 
SURVEY_PERIOD    = 60.0
METABOLIC_PERIOD = 180.0
REF_POINT        = plr.reference_point(
    bounds   = [[2.0, -1.0], [6.0, 5.0]],
    maximize = [MAXIMIZE[METABOLIC], MAXIMIZE[COMFORT]],
    margin   = 0.1
)

OUTPUT_DIR       = Path('hilo/output/experiments') / time.strftime('%Y%m%d_%H%M%S')

@dataclass
class Config:
    gp_noise          : plr.NoiseModel
    min_lengthscale   : float
    num_queries       : int
    repeats           : dict[str, int]
    dim               : int = DIM
    seed              : int = SEED

def get_metabolic_data(action, trial_num):
    time.sleep(METABOLIC_PERIOD)
    for _ in range(3):
        plr.chime()
    while True:
        ret = logged_input('Please enter metabolic cost (W/kg): ')
        try:
            ret = float(ret)
        except ValueError:
            logger.error(f'{ret} not interpretable by Python float. Try again please.',
                         extra=TO_BOTH)
            continue
            
        ans = logged_input(f'Got {ret} W/kg. Continue? (y/n)? ')
        if ans.lower() == 'y':
            break
    
    return ret

def make_probes_mo(ipad: Survey, multithread=False):
    """One probe per objective, each measuring its own column of the MO truth.
    """
    probes = [
        plr.Probe(
            name              = METABOLIC,
            caller            = get_metabolic_data,
            repeats           = REPEATS[METABOLIC],
            obj_name          = METABOLIC,
            separate_thread   = multithread
        ),
        plr.Probe(
            name              = COMFORT,
            caller            = ipad.ask,
            repeats           = REPEATS[COMFORT],
            obj_name          = COMFORT,
            separate_thread   = multithread
        ),
    ]
    return probes

def make_experiment_mo(
    probes  : list[plr.Probe],
    maximize: dict[str, bool]
):
    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(
                name          = name,
                maximize      = maximize[name],
                action_bounds = BOUNDS
            )
            for name in (METABOLIC, COMFORT)
        ],
        probes       = probes,
        action_names = ACTION_NAMES
    )
    return experiment