import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
warnings.filterwarnings('ignore', category=NumericalWarning)

DIM              = 3
BOX              = 5.0
SEED             = 95
TRUE_NOISE       = 0.1 #0.25
NUM_QUERIES      = DIM * 13
ACQ_STRATS       = ['ucb', 'logei', 'qlognei', 'ts']
REPEATS          = 1
COMFORT          = 'Comfort'
METABOLIC        = 'Metabolic Cost'
MULTITHREAD      = False

RUNS_PER_ACQF = 5

OUTPUT_DIR       = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS       = {}   # acquisition knobs overriding acquisition_factory_1d's own

GROUND_TRUTH_1D = plr.SyntheticOracleParams(
    func            = 'Levy',
    objectives      = (COMFORT,),
    dim             = DIM,
    box             = BOX,
    seed            = SEED,
    rel_noise_std   = TRUE_NOISE,
)

GROUND_TRUTH_2D = plr.SyntheticOracleParams(
    func          = 'BraninCurrin',
    objectives    = (METABOLIC, COMFORT),
    dim           = 2,
    box           = None,     # BraninCurrin takes no bounds; its own are [0, 1]^2
    seed          = SEED,
    rel_noise_std = TRUE_NOISE
)

COMFORT_TRUTH   = GROUND_TRUTH_1D.build()
MO_TRUTH        = GROUND_TRUTH_2D.build()

# read off the truth rather than restated, so the action frame the GP is fit in
# cannot drift from the domain the groundtruth is defined on
MO_DIM          = MO_TRUTH.objectives[0].truth.dim
MO_BOUNDS       = plr.as_bounds(MO_TRUTH.objectives[0].truth.bounds)

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
            name              = COMFORT,
            caller            = COMFORT_TRUTH,
            repeats           = REPEATS,
            obj_name          = COMFORT,
            separate_thread   = MULTITHREAD
        ),
    ]
    return probes


def make_probes_mo():
    """One probe per objective, each measuring its own column of the MO truth.
    """
    probes = [
        plr.Probe(
            name              = name,
            caller            = MO_TRUTH.objective(i),
            repeats           = REPEATS,
            obj_name          = name,
            separate_thread   = MULTITHREAD
        )
        for i, name in enumerate(GROUND_TRUTH_2D.objectives)
    ]
    return probes


def make_experiment_1d(probes: list[plr.Probe]):
    experiment = plr.Logger(
        objectives   = [
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

def make_experiment_mo(
    probes: list[plr.Probe],
    maximize: dict[str, bool]
):
    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(
                name          = name,
                maximize      = maximize[name],
                action_bounds = MO_BOUNDS
            )
            for name in GROUND_TRUTH_2D.objectives
        ],
        probes       = probes,
        device       = plr.Device(),
        action_names = [f'x{i}' for i in range(MO_DIM)],
    )
    return experiment