import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
from tqdm import tqdm
import hilo.shared.simulation as hilo
import scripts.mo_example as mo
warnings.filterwarnings('ignore', category=NumericalWarning)

OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
SCAN_SAMPLES  = 8192    # Sobol points in the groundtruth scan the fronts are read off

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

class OracleProblem(Problem):

    def __init__(
        self,
        oracle  : plr.MOSyntheticOracle,
        signs   : np.ndarray,
        noise   : bool = True
    ):
        bounds = plr.as_bounds(oracle.get_oracle(0).truth.bounds)
        super().__init__(
            n_var = bounds.shape[1],
            n_obj = len(oracle),
            xl    = bounds[0],
            xu    = bounds[1]
        )
        self.oracle   = oracle
        self.signs    = np.asarray(signs, dtype=float)
        self.noise    = noise
        self.queried  = []
        self.measured = []

    def _evaluate(self, X, out, *args, **kwargs):
        """The (n, d) population X -> its (n, m) values, in minimization space."""
        values = self.oracle(X, noise=self.noise)
        self.queried.append(np.asarray(X, dtype=float))
        self.measured.append(values)
        out['F'] = -self.signs * values


def run_nsga2(
    ground_truth    : plr.MOSyntheticOracle,
    objectives      : plr.DecoupledObjectives,
    num_queries     : int,
    pop_size        : int = 5,
    noise           : bool = True,
    seed            : int = 95
):
    problem = OracleProblem(
        oracle  = ground_truth,
        signs   = objectives.signs,
        noise   = noise
    )
    minimize(
        problem     = problem,
        algorithm   = NSGA2(pop_size=pop_size),
        termination = ('n_eval', num_queries),
        seed        = seed,
        verbose     = False
    )

    return np.vstack(problem.queried), np.vstack(problem.measured)


def aux(
    recommended     : np.ndarray,
    queried         : np.ndarray,
    objectives      : plr.DecoupledObjectives,
    ground_truth    : plr.MOSyntheticOracle
):
    return {
        'hv_regret'          : plr.normalized_hypervolume_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = objectives,
            ref_point       = None
        ),
        'hv_regret_attained' : plr.attained_hypervolume_regret(
            raw_actions     = queried,
            ground_truth    = ground_truth,
            objectives      = objectives,
            ref_point       = None
        ),
        'front_alignment'    : plr.front_alignment_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = objectives
        ),
        'front_coverage'     : plr.front_coverage_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = objectives
        )
    }


def nsga2_metrics(
    queried         : np.ndarray,
    measured        : np.ndarray,
    objectives      : plr.DecoupledObjectives,
    ground_truth    : plr.MOSyntheticOracle,
    pop_size        : int,
    show_progress   : bool = False
):
    values = objectives.maximization_space(measured)

    # pymoo evaluates whole populations, so N is a multiple of pop_size; a
    # partial tail still gets a point rather than being dropped
    evals = list(range(pop_size, len(queried) + 1, pop_size))
    if not evals or evals[-1] != len(queried):
        evals.append(len(queried))

    curves = {}
    for n in tqdm(evals, disable=not show_progress):
        prefix = queried[:n]
        front  = plr.get_nondominated(values[:n])
        for name, value in aux(prefix[front], prefix, objectives,
                               ground_truth).items():
            curves.setdefault(name, []).append(value)

    return (np.asarray(evals),
            {name: np.asarray(curve) for name, curve in curves.items()})

def make_idealpoint_groundtruth(
    seed, 
    box, 
    noise, 
    dim,
    num_objs,
    low=None, 
    high=None,
):
    box       = np.asarray(box)
    rng       = np.random.default_rng(seed)
    optima    = rng.random(size=(num_objs, dim)) * (box[1] - box[0]) + box[0]
    weights   = 0.9 * rng.random(size=(num_objs,)) + 0.1
    GROUND_TRUTH_PARAMS = plr.SyntheticOracleParams(
        func          = 'IdealPoint',
        objectives    = [f'$f_{i}' for i in range(num_objs)],
        optima        = optima,
        weights       = weights,
        offset        = None,
        low           = low,
        high          = high,
        dim           = dim,
        box           = box,
        seed          = seed,
        rel_noise_std = noise,
        measure       = 'std',
        n_spread      = SCAN_SAMPLES
    )

    gt = GROUND_TRUTH_PARAMS.build()
    return gt, GROUND_TRUTH_PARAMS


def run_nsga_trial(
    seed: int, 
    pop_size: int, 
    groundtruth: plr.MOSyntheticOracle,
    bounds: np.ndarray,
    maximize: bool,
    num_objs: int,
    num_queries: int
):

    objectives = plr.DecoupledObjectives([
        plr.Objective.from_empty(
            name          = f'$f_{i}$',
            maximize      = maximize,
            action_bounds = bounds
        )
        for i in range(num_objs)
    ])

    queried, measured = run_nsga2(
        ground_truth = groundtruth,
        objectives   = objectives,
        num_queries  = num_queries,
        pop_size     = pop_size,
        seed         = seed
    )
    evals, curves = nsga2_metrics(
        queried      = queried,
        measured     = measured,
        objectives   = objectives,
        ground_truth = groundtruth,
        pop_size     = pop_size
    )
    return evals, curves, queried, measured

def make_probes_mo(gt: plr.MOSyntheticOracle, num_objs: int):
    """One probe per objective, each measuring its own column of the MO truth.
    """
    probes = [
        plr.Probe(
            name              = f'$f_{i}$',
            caller            = lambda action, i=i: gt.get_oracle(i)(action),
            repeats           = 1,
            obj_name          = f'$f_{i}$',
            separate_thread   = False
        ) for i in range(num_objs)
    ]
    return probes

def make_experiment_mo(
    probes: list[plr.Probe],
    bounds: np.ndarray,
    num_objs: int,
    maximize: bool,
    action_names: str = None
):
    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(
                name          = f'$f_{i}$',
                maximize      = maximize,
                action_bounds = bounds
            )
            for i in range(num_objs)
        ],
        probes       = probes,
        action_names = action_names
    )
    return experiment

def setup_experiment(
    acq_strat: str, 
    seed: int, 
    gt: plr.SyntheticOracle,
    num_objs: int,
    bounds: np.ndarray,
    maximize: bool,
    action_names: list[str]
):
    probes     = make_probes_mo(
        gt       = gt,
        num_objs = num_objs
    )
    experiment = make_experiment_mo(
        probes       = probes,
        bounds       = bounds,
        maximize     = maximize,
        num_objs     = num_objs,
        action_names = action_names
    )
    params     = plr.AcquisitionParams(
        strategy          = acq_strat,
        seed              = seed,
        num_objectives    = num_objs,
        raw_ref_point     = None,
    )
    # assert experiment.objectives.names == list(gt.objectives)

    return experiment, params

def gp_aux(
    gp              : plr.DecoupledMOGP,
    ground_truth    : plr.MOSyntheticOracle,
    num_sparse      : int,
    rng             : np.random.Generator
):
    queried = np.vstack([o.xdata for o in gp.objectives.objectives])

    mu, _  = gp.posterior_at(ground_truth.scan_actions, raw=True)
    front  = plr.get_nondominated(gp.objectives.maximization_space(mu))
    sparse = rng.choice(front, size=min(num_sparse, len(front)), replace=False)

    metrics = {}
    for prefix, idx in (('', front), ('sparse_', sparse)):
        for name, value in aux(ground_truth.scan_actions[idx], queried,
                               gp.objectives, ground_truth).items():
            metrics[prefix + name] = value

    return metrics

def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    ground_truth      : plr.MOSyntheticOracle,
    gp_noise          : plr.NoiseModel,
    num_objs          : int,
    num_random=3,
    num_queries=45,
    dim=3,
    seed=95,
    min_lengthscale=None,
):
    gp = None
    rng = np.random.default_rng(seed)
    for i in tqdm(range(num_queries)):
        if i < num_random:
            # randomly sample until the acquisition has something to fit to
            source = 'random'
            action = plr.sample_actions(
                dim    = dim,
                n      = 1,
                kind   = 'uniform',
                seed   = seed + i,
                bounds = experiment.objectives.action_bounds
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition.strategy
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action = action,
            args           = {
                f'$f_{i}$': (action,)
            for i in range(num_objs)
            } 
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

        gp = plr.DecoupledMOGP(
            objectives          = experiment.objectives,
            noise               = gp_noise,
            fit_hyperparameters = True,
            min_length_scale    = min_lengthscale,
        )
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
            aux         = gp_aux(gp, ground_truth, num_queries, rng)
        )

    return experiment, gp, dataset

def run_mobo_trial(
    seed, 
    groundtruth: plr.SyntheticOracle, 
    gt_params: plr.SyntheticOracleParams,
    num_objs: int,
    bounds: np.ndarray,
    maximize: bool,
    num_queries: int,
    gp_noise: plr.NoiseModel,
    num_random: int,
    dim: int,
    action_names: str = None,
    min_lengthscale: float = None,
    output_dir: Path = OUTPUT_DIR,
):
    experiment, params = setup_experiment(
        acq_strat   = 'qlognparego',
        seed        = seed,
        gt          = groundtruth,
        num_objs = num_objs,
        bounds = bounds,
        maximize = maximize,
        action_names = action_names,
    )

    dataset = plr.ExperimentDataset(
        name         = 'qlognparego',
        acquisition  = params,
        groundtruth  = gt_params,
        config       = mo.asdict(
            hilo.Config(
                gp_noise          = plr.NoiseModel.coerce(gp_noise),
                min_lengthscale   = min_lengthscale,
                num_queries       = num_queries,
                repeats           = None,
                dim               = bounds.shape[1],
                seed              = seed
            )
        ),
        path         = output_dir / f'{seed}.json'
    )

    experiment, gp, dataset = run_experiment(
        experiment        = experiment,
        acqf              = params.build(),
        dataset           = dataset,
        ground_truth      = groundtruth,
        gp_noise          = gp_noise,
        num_objs=num_objs,
        num_random=num_random,
        num_queries=num_queries,
        dim=dim,
        seed=seed,
        min_lengthscale=min_lengthscale
    )
    print(f'wrote {dataset.save()}')

    dataset: plr.ExperimentDataset = dataset
    names  = ('front_coverage', 'front_alignment', 'hv_regret', 'hv_regret_attained')
    evals  = np.arange(len(dataset)) + 1
    dense  = {n: dataset.get_aux(n) for n in names} | {'evals': evals}
    sparse = {n: dataset.get_aux(f'sparse_{n}') for n in names} | {'evals': evals}
    return dense, sparse


def save_trial(
    output_dir  : Path,
    trial       : int,
    nsga2       : dict,
    queried     : np.ndarray,
    measured    : np.ndarray,
    dense       : dict,
    sparse      : dict
):
    """Writes one trial's curves as `{method}_trial{trial}.csv`, and NSGA-II's
    (N, d) queried actions and (N, m) noisy values as `NSGA2_trial{trial}.npz`."""
    for method, curves in (('NSGA2', nsga2), ('MOBO-dense', dense),
                           ('MOBO-sparse', sparse)):
        pd.DataFrame.from_dict(curves).to_csv(output_dir / f'{method}_trial{trial}.csv')
    np.savez(output_dir / f'NSGA2_trial{trial}.npz', queried=queried,
             measured=measured)


def main():
    SAVE_DATA = True
    if not SAVE_DATA:
        print('WARNING NOT SAVING DATA')
    if SAVE_DATA: OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIM = 3
    NUM_OBJS = 2
    BOX = (-3,3)
    BOUNDS = plr.as_bounds(BOX, dim=DIM)
    TRUE_NOISE = 0.1
    GUESS_FRAC = 0.3
    SEED = 95
    LOW = (-1,) * NUM_OBJS
    HIGH = (1,) * NUM_OBJS
    MAXIMIZE = False
    NUM_QUERIES = 45
    NUM_RANDOM = 3
    ACTION_NAMES = [f'$x_{i}$' for i in range(DIM)]
    MIN_LENGTHSCALE = 0.1
    POP_SIZE      = 5   
    N_TRIALS      = 5

    for i in tqdm(range(N_TRIALS)):
        torch.manual_seed(SEED + i)
        NOISE_ESTIMATE = np.clip(1e-3, None, TRUE_NOISE + np.random.random() * GUESS_FRAC * TRUE_NOISE)
        GP_NOISE   = plr.NoiseModel.prior(NOISE_ESTIMATE)
        gt, params = make_idealpoint_groundtruth(
            seed     = SEED + i,
            box      = BOX,
            noise    = TRUE_NOISE,
            dim      = DIM,
            low      = LOW,
            high     = HIGH,
            num_objs = NUM_OBJS
        )

        evals, curves, queried, measured = run_nsga_trial(
            seed        = SEED + i,
            pop_size    = POP_SIZE,
            groundtruth = gt,
            bounds      = BOUNDS,
            maximize    = MAXIMIZE,
            num_objs    = NUM_OBJS,
            num_queries = 90 #NUM_QUERIES,
        )

        curves['evals'] = evals

        dense, sparse = run_mobo_trial(
            seed        = SEED + i,
            groundtruth = gt,
            gt_params   = params,
            num_objs=NUM_OBJS,
            bounds=BOUNDS,
            maximize=MAXIMIZE,
            num_queries=NUM_QUERIES,
            gp_noise=GP_NOISE,
            action_names=ACTION_NAMES,
            min_lengthscale=MIN_LENGTHSCALE,
            num_random=NUM_RANDOM,
            dim=DIM
        )
        if SAVE_DATA:
            save_trial(OUTPUT_DIR, i, curves, queried, measured, dense, sparse)


if __name__ == '__main__':
    main()
