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
ACQ_STRATS    = ['qlognehvi', 'qlognparego', 'qhvkg', 'qlogehvi']
RUNS_PER_ACQF = 3
OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
ACQ_KWARGS    = {}   # acquisition knobs overriding acquisition_factory_1d's own
POP_SIZE      = 9    # NSGA2 spends this many evaluations per generation


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

class OracleProblem(Problem):
    """pymoo's minimization view of a `MOSyntheticOracle`.

    pymoo minimizes every column, and the oracle reports the truth's own raw
    values, so `F` is `-signs * raw`: `signs` maps raw onto maximization space
    and the negation puts it back into pymoo's.

    Args:
        oracle: the `MOSyntheticOracle` evaluated.
        signs: (m,) `DecoupledObjectives.signs`, one per truth column.
        noise: whether an evaluation is measured or noiseless.

    Attributes:
        queried: the (pop_size, d) blocks handed to `_evaluate`, in order.
        measured: the (pop_size, m) values seen there, in the truth's raw units.
    """

    def __init__(
        self,
        oracle  : plr.MOSyntheticOracle,
        signs   : np.ndarray,
        noise   : bool = True
    ):
        bounds = plr.as_bounds(oracle.objectives[0].truth.bounds)
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
    """NSGA2 on the truth, budget-matched to an acquisition run.

    The budget is stated in evaluations rather than generations, since that is
    what an acquisition run spends: `pop_size` of them go on the initial
    population before a single generation, so a population near the whole
    budget never breeds.

    Args:
        ground_truth: the oracle NSGA2 optimizes, and the metric scores against.
        objectives: the `DecoupledObjectives` whose `signs` fix the directions.
        num_queries: the evaluation budget. pymoo checks the termination once
            per generation, so a `pop_size` not dividing it overshoots by less
            than one population.
        pop_size: individuals per generation.
        noise: whether NSGA2 measures the truth or sees it noiselessly. An
            acquisition run measures, so a comparison wants True; NSGA2 keeps
            each individual's single draw for as long as it survives.
        seed: seeds pymoo's sampling and variation.

    Returns:
        `(queried, measured)`: the (N, d) actions evaluated, in order, and the
        (N, m) values seen there.
    """
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
    gp              : plr.DecoupledMOGP,
    ground_truth    : plr.MOSyntheticOracle
):
    queried = np.vstack([gp.objectives[i].xdata for i in range(len(gp.objectives))])

    return {
        'hv_regret'          : plr.normalized_hypervolume_regret(
            gp              = gp,
            ground_truth    = ground_truth
        ),
        'hv_regret_attained' : plr.attained_hypervolume_regret(
            raw_actions     = queried,
            ground_truth    = ground_truth,
            objectives      = gp.objectives
        ),
        'front_alignment'    : plr.front_alignment_regret(
            gp              = gp,
            ground_truth    = ground_truth
        )
    }


def nsga2_metrics(
    queried         : np.ndarray,
    measured        : np.ndarray,
    objectives      : plr.DecoupledObjectives,
    ground_truth    : plr.MOSyntheticOracle
):
    """`aux` over every prefix of an NSGA2 run.

    NSGA2 carries no posterior, so `normalized_hypervolume_regret` and
    `front_alignment_regret` are undefined for it as it stands. Fitting a
    `DecoupledMOGP` to the points it queried is what makes them apply: they then
    score the *design* NSGA2 produced, under the same model, noise and
    lengthscale floor an acquisition run scores its own with.

    The GP is fit to `measured` rather than to a fresh evaluation of the truth,
    so it sees the draws NSGA2 itself bred on.

    Args:
        queried: (N, d) actions, in evaluation order.
        measured: (N, m) values seen there, in the truth's raw units.
        objectives: the template each prefix is rebuilt from - names,
            directions and action bounds.
        ground_truth: the oracle every metric scores against.

    Returns:
        name -> (N,) curve, one entry per key of `aux`.
    """
    curves = {}
    for k in tqdm(range(len(queried))):
        prefix = plr.DecoupledObjectives([
            plr.Objective.from_data(
                actions       = queried[:k + 1],
                values        = measured[:k + 1, j],
                maximize      = objective.maximize,
                name          = objective.name,
                action_bounds = objective.action_bounds
            )
            for j, objective in enumerate(objectives.objectives)
        ])
        gp = plr.DecoupledMOGP(
            objectives          = prefix,
            noise               = hilo.GP_NOISE,
            fit_hyperparameters = True,
            min_length_scale    = hilo.MIN_LENGTHSCALE
        )
        for name, value in aux(gp, ground_truth).items():
            curves.setdefault(name, []).append(value)

    return {name: np.asarray(curve) for name, curve in curves.items()}


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
        raw_ref_point     = hilo.MO_TRUTH.ref_point,
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

    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # torch.manual_seed(hilo.SEED)

    # for acq_strat in ACQ_STRATS:
    #     for trial in range(RUNS_PER_ACQF):
    #         run_trial(acq_strat, trial)

    # print('Done')

    objectives = hilo.make_experiment_mo(
        probes   = hilo.make_probes_mo(),
        maximize = hilo.MAXIMIZE
    ).objectives

    queried, measured = run_nsga2(
        ground_truth = hilo.MO_TRUTH,
        objectives   = objectives,
        num_queries  = hilo.NUM_QUERIES,
        pop_size     = 100
    )
    curves = nsga2_metrics(
        queried      = queried,
        measured     = measured,
        objectives   = objectives,
        ground_truth = hilo.MO_TRUTH
    )
    for name, curve in curves.items():
        print(f'{name:<20}: {curve[0]:.4f} -> {curve[-1]:.4f}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evals = np.arange(1, len(queried) + 1)
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.5))
    for ax, (name, curve) in zip(axes, curves.items()):
        # NSGA2 selects once per population, so every point between two
        # boundaries was bred from the same parents
        for boundary in range(POP_SIZE, len(evals), POP_SIZE):
            ax.axvline(boundary + 0.5, color='0.85', lw=0.8, zorder=0)
        ax.plot(evals, curve, marker='.')
        ax.set(xlabel='evaluations', ylabel=name)
    fig.suptitle(f'NSGA2 (pop_size={POP_SIZE}) on {hilo.GT_NAME}')
    fig.tight_layout()
    path = OUTPUT_DIR / 'nsga2_regret.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')

if __name__ == '__main__':
    main()
