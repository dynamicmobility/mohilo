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
warnings.filterwarnings('ignore', category=NumericalWarning)

OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
POP_SIZE      = 5    # NSGA2 spends this many evaluations per generation
N_TRIALS      = 50

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
    pop_size        : int = POP_SIZE,
    noise           : bool = True,
    seed            : int = hilo.SEED
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
    pop_size        : int = POP_SIZE,
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

def make_idealpoint_groundtruth(seed, bounds, noise):
    bounds    = np.asarray(bounds)
    rng       = np.random.default_rng(seed)
    optima    = rng.random(size=(2,3)) * (bounds[1] - bounds[0]) + bounds[0]
    GROUND_TRUTH_PARAMS = plr.SyntheticOracleParams(
        func          = 'IdealPoint',
        objectives    = ('Metabolic Cost', 'Comfort'),
        optima        = optima,
        dim           = 3,
        box           = bounds,
        seed          = seed,
        rel_noise_std = noise,
        measure = 'std'
    )

    gt = GROUND_TRUTH_PARAMS.build()
    return gt


def run_nsga_trial(seed: int, pop_size: int, groundtruth: plr.MOSyntheticOracle):

    objectives = hilo.make_experiment_mo(
        probes   = hilo.make_probes_mo(),
        maximize = hilo.MAXIMIZE
    ).objectives

    queried, measured = run_nsga2(
        ground_truth = groundtruth,
        objectives   = objectives,
        num_queries  = hilo.NUM_QUERIES,
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
    return evals, curves

def run_mobo(seed, groundtruth):
    pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(N_TRIALS)):
        gt = make_idealpoint_groundtruth(
            seed    = hilo.SEED + i,
            bounds  = (-3.0, 3.0),
            noise   = 0.1
        )
        evals, curves = run_nsga_trial(
            seed        = hilo.SEED + i,
            pop_size    = POP_SIZE,
            groundtruth = gt
        )
        curves['evals'] = evals
        pd.DataFrame.from_dict(curves).to_csv(
            OUTPUT_DIR / f'NSGA2_trial{i}.csv'
        )


if __name__ == '__main__':
    main()
