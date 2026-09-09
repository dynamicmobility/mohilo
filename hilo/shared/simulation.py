import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
warnings.filterwarnings('ignore', category=NumericalWarning)

MULTITHREAD = False 

# Objectives
NUM_OBJECTIVES   = 2
COMFORT          = 'Comfort'
METABOLIC        = 'Metabolic Cost'
REPEATS          = {METABOLIC: 1, COMFORT: 1}
MAXIMIZE         = {METABOLIC: False, COMFORT: False}
SURVEY_TIMEOUT   = 0.0
SURVEY_PERIOD    = 0.0
METABOLIC_PERIOD = 0.0
REF_MARGIN       = 0.1  # reference sits this fraction of the front's extent past its nadir
TRUE_NOISE       = 0.01

# Optimization
SEED             = 95
NUM_QUERIES      = 45
ACQ_STRAT        = 'qlognehvi'
ACQ_KWARGS       = {}
GP_NOISE         = plr.NoiseModel.prior(TRUE_NOISE)
# GP_NOISE         = plr.NoiseModel.prior(0.01)
MIN_LENGTHSCALE  = 0.05
NUM_RANDOM       = 3


GT_NAME = 'IdealPoint'
# one quadratic bowl per objective, both on the box `box` states -- the shared
# box is what keeps both optima inside the action space, since
# MOSyntheticOracle.bounds intersects its members' boxes
GROUND_TRUTH_PARAMS = plr.SyntheticOracleParams(
    func          = GT_NAME,
    objectives    = (METABOLIC, COMFORT),
    optima        = ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    dim           = 3,
    box           = (0.0, 3.0),
    seed          = SEED,
    rel_noise_std = TRUE_NOISE,
    measure = 'std'
)

MO_TRUTH        = GROUND_TRUTH_PARAMS.build()


REF_POINT       = plr.reference_point(
    values   = MO_TRUTH.scan_values,
    maximize = [MAXIMIZE[name] for name in GROUND_TRUTH_PARAMS.objectives],
    margin   = REF_MARGIN
)

# recorded on the params, so a metric reading a saved run back scores it
# against the reference the run optimized against
GROUND_TRUTH_PARAMS = replace(GROUND_TRUTH_PARAMS, ref_point=tuple(REF_POINT))

# Actions
DIM             = MO_TRUTH.get_oracle(0).truth.dim
BOUNDS          = plr.as_bounds(MO_TRUTH.get_oracle(0).truth.bounds)
ACTION_NAMES     = [f'x{i}' for i in range(DIM)]
OUTPUT_DIR       = Path('hilo/output/experiments') / time.strftime('%Y%m%d_%H%M%S')

@dataclass
class Config:
    gp_noise          : plr.NoiseModel
    min_lengthscale   : float
    num_queries       : int
    repeats           : dict[str, int]
    dim               : int = DIM
    seed              : int = SEED


def make_probes_mo(ipad=None, multithread=False):
    """One probe per objective, each measuring its own column of the MO truth.
    """
    probes = [
        plr.Probe(
            name              = METABOLIC,
            caller            = lambda action, trial_num: MO_TRUTH.get_oracle(0)(action),
            repeats           = REPEATS[METABOLIC],
            obj_name          = METABOLIC,
            separate_thread   = multithread
        ),
        plr.Probe(
            name              = COMFORT,
            caller            = lambda action, trial, timeout, period: MO_TRUTH.get_oracle(1)(action),
            repeats           = REPEATS[COMFORT],
            obj_name          = COMFORT,
            separate_thread   = multithread
        )
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