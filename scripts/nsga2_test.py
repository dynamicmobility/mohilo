import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
from tqdm import tqdm
import hilo.shared.simulation as hilo
warnings.filterwarnings('ignore', category=NumericalWarning)

# TODO: go through all the reference setting/computing and min/max objective logic in this codebase
OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')
POP_SIZE      = 9    # NSGA2 spends this many evaluations per generation

# the three front colors, validated as a scatter palette (all-pairs CVD dE 9.2)
TRUE_FRONT    = '#2a78d6'
NSGA_MEASURED = '#1baf7a'
NSGA_ATTAINED = '#eb6834'


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
            ref_point       = hilo.REF_POINT
        ),
        'hv_regret_attained' : plr.attained_hypervolume_regret(
            raw_actions     = queried,
            ground_truth    = ground_truth,
            objectives      = objectives,
            ref_point       = hilo.REF_POINT
        ),
        'front_alignment'    : plr.front_alignment_regret(
            raw_actions     = recommended,
            ground_truth    = ground_truth,
            objectives      = objectives
        ),
        # precision and coverage read opposite ways: a tight front scores well
        # on alignment and badly here, a broad one with strays the other way
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
    pop_size        : int = POP_SIZE
):
    values = objectives.maximization_space(measured)

    # pymoo evaluates whole populations, so N is a multiple of pop_size; a
    # partial tail still gets a point rather than being dropped
    evals = list(range(pop_size, len(queried) + 1, pop_size))
    if not evals or evals[-1] != len(queried):
        evals.append(len(queried))

    curves = {}
    for n in tqdm(evals):
        prefix = queried[:n]
        front  = plr.get_nondominated(values[:n])
        for name, value in aux(prefix[front], prefix, objectives,
                               ground_truth).items():
            curves.setdefault(name, []).append(value)

    return (np.asarray(evals),
            {name: np.asarray(curve) for name, curve in curves.items()})


def plot_front(
    ax,
    queried         : np.ndarray,
    measured        : np.ndarray,
    objectives      : plr.DecoupledObjectives,
    ground_truth    : plr.MOSyntheticOracle,
    ref_point       : np.ndarray = None
):
    """The truth's own Pareto front and the one NSGA2 recommends, in objective
    space. Two objectives only, since that is what `plot_pareto` draws.

    NSGA2's front is drawn twice, because the two are what the noise separates:
    the values it *measured* are what it picked the front on, and the truth at
    those same actions is what the metrics score it as. A gap between them is
    the observation noise, not a modelling error.

    Args:
        queried, measured: the (N, d) actions and (N, m) values NSGA2 saw.
        objectives: the directions each column is read in.
        ground_truth: the oracle, whose scan draws the reachable set and the
            true front.
        ref_point: (m,) the hypervolume reference, marked when it falls inside
            the view. The axes are set by the fronts, since a reference sits
            past the worst end of the scan by construction and letting it fix
            the scale squashes every front into a corner.

    Returns:
        The `ax`.
    """
    if objectives.num_objectives != 2:
        raise ValueError(f'a front plot is 2D, got {objectives.num_objectives} '
                         'objectives')

    scan     = ground_truth.scan_values
    true_idx = plr.get_nondominated(objectives.maximization_space(scan))

    front    = plr.get_nondominated(objectives.maximization_space(measured))
    attained = ground_truth(queried[front], noise=False)

    # one color per row, so the scan's dominated points read as the reachable
    # set behind the front rather than as more of it
    scan_colors           = np.tile(to_rgba('0.85'), (len(scan), 1))
    scan_colors[true_idx] = to_rgba(TRUE_FRONT)

    plr.plot_pareto(
        ax             = ax,
        pareto         = scan,
        nd_idx         = true_idx,
        colors         = scan_colors,
        connect        = True,
        dominated_s    = 6,
        nondominated_s = 42,
        label          = f'true front ({len(true_idx)})',
        set_lims       = False
    )

    # shapes as well as hues, so the three sets stay apart without color
    for values, color, marker, label in (
        (measured[front], NSGA_MEASURED, '^', f'NSGA2 front, measured ({len(front)})'),
        (attained,        NSGA_ATTAINED, 'X', 'NSGA2 front, truth at those actions')
    ):
        plr.plot_pareto(
            ax             = ax,
            pareto         = values,
            nd_idx         = np.arange(len(values)),
            colors         = color,
            show_dominated = False,
            nondominated_s = 58,
            label          = label,
            set_lims       = False,
            marker         = marker
        )

    # the fronts, not the reference or the whole reachable set, set the scale.
    # plot_pareto's own set_lims is multiplicative, which inverts on the
    # negative values a noisy measurement produces here
    focus  = np.vstack([scan[true_idx], measured[front], attained])
    lo, hi = focus.min(axis=0), focus.max(axis=0)
    pad    = 0.10 * np.where(hi > lo, hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad

    if ref_point is not None:
        ref = np.asarray(ref_point, dtype=float)
        if np.all((ref >= lo) & (ref <= hi)):
            ax.scatter(*ref, s=70, marker='*', facecolor='none',
                       edgecolor='0.35', linewidths=1.2, zorder=6)
            ax.annotate('reference', ref, textcoords='offset points',
                        xytext=(-6, -12), ha='right', fontsize=8, color='0.35')

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    # the reachable set is plot_pareto's dominated half, which carries no label
    ax.scatter([], [], s=36, c='0.85', label='reachable (truth scan)')

    arrow = {True: '\u2191', False: '\u2193'}
    ax.set_xlabel(f'{objectives.objectives[0].name} '
                  f'({arrow[objectives.objectives[0].maximize]})', fontsize=11)
    ax.set_ylabel(f'{objectives.objectives[1].name} '
                  f'({arrow[objectives.objectives[1].maximize]})', fontsize=11)
    ax.grid(color='0.92', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    legend = ax.legend(loc='upper right', fontsize=8, frameon=False)
    for handle in legend.legend_handles:
        # the reachable cloud is drawn small enough to vanish in the key
        handle.set_sizes([36])

    return ax


def main():

    objectives = hilo.make_experiment_mo(
        probes   = hilo.make_probes_mo(),
        maximize = hilo.MAXIMIZE
    ).objectives

    pop_size = 5   #100
    queried, measured = run_nsga2(
        ground_truth = hilo.MO_TRUTH,
        objectives   = objectives,
        num_queries  = hilo.NUM_QUERIES,
        pop_size     = pop_size,
    )
    evals, curves = nsga2_metrics(
        queried      = queried,
        measured     = measured,
        objectives   = objectives,
        ground_truth = hilo.MO_TRUTH,
        pop_size     = pop_size
    )
    for name, curve in curves.items():
        print(f'{name:<20}: {curve[0]:.4f} -> {curve[-1]:.4f}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.5))
    for ax, (name, curve) in zip(axes, curves.items()):
        # every point is a generation boundary, which is the only state NSGA2
        # actually occupies
        ax.plot(evals, curve, marker='.')
        ax.set(xlabel='evaluations', ylabel=name)
    fig.suptitle(f'NSGA2 (pop_size={pop_size}, {len(evals)} generations) '
                 f'on {hilo.GT_NAME}')
    fig.tight_layout()
    path = OUTPUT_DIR / 'nsga2_regret.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_front(
        ax           = ax,
        queried      = queried,
        measured     = measured,
        objectives   = objectives,
        ground_truth = hilo.MO_TRUTH,
        ref_point    = hilo.REF_POINT
    )
    ax.set_title(f'NSGA2 (pop_size={pop_size}, {len(queried)} evaluations) '
                 f'on {hilo.GT_NAME}')
    fig.tight_layout()
    path = OUTPUT_DIR / 'nsga2_front.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')

if __name__ == '__main__':
    main()
