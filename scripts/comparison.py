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
warnings.filterwarnings('ignore', category=NumericalWarning)

OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
POP_SIZE      = 5    # NSGA2 spends this many evaluations per generation
N_TRIALS      = 3

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
    return gt, GROUND_TRUTH_PARAMS


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
    return evals, curves, queried, measured

def make_probes_mo(gt, ipad=None, multithread=False):
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
    for i in tqdm(range(N_TRIALS)):
        hilo.BOUNDS     = (-3, 3)
        hilo.TRUE_NOISE = 0.1
        hilo.GP_NOISE   = plr.NoiseModel.prior(hilo.TRUE_NOISE)

        gt, params = make_idealpoint_groundtruth(
            seed    = hilo.SEED + i,
            bounds  = hilo.BOUNDS,
            noise   = hilo.TRUE_NOISE
        )
        evals, curves, queried, measured = run_nsga_trial(
            seed        = hilo.SEED + i,
            pop_size    = POP_SIZE,
            groundtruth = gt,
        )

        curves['evals'] = evals
        pd.DataFrame.from_dict(curves).to_csv(
            OUTPUT_DIR / f'NSGA2_trial{i}.csv'
        )
        # (N, d) actions and (N, m) noisy values, in query order
        np.savez(OUTPUT_DIR / f'NSGA2_trial{i}.npz', queried=queried,
                 measured=measured)

        curves = run_mobo_trial(
            seed        = hilo.SEED + i,
            groundtruth = gt,
            gt_params   = params
        )
        pd.DataFrame.from_dict(curves).to_csv(
            OUTPUT_DIR / f'MOBO_trial{i}.csv'
        )


if __name__ == '__main__':
    main()
