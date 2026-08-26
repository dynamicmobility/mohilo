"""BoTorch synthetic test functions as groundtruths: the registry the
experiments draw from, noiseless evaluation, and an observation noise stated as
a fraction of a function's own spread."""

from dataclasses import dataclass
from inspect import signature

import numpy as np
import torch

from botorch.exceptions.errors import BotorchError
from botorch.test_functions import SyntheticTestFunction, multi_objective, synthetic
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)

from pypolar.optimization.gp import DTYPE
from pypolar.optimization.objectives import sample_actions
from pypolar.utils.pareto import get_nondominated, hypervolume_from_nondominated

SPREAD_SAMPLES = 4096   # Sobol points a spread is measured over
PROBE_SAMPLES  = 32     # Sobol points a candidate instance is probed at

SYNTHETIC_FUNCTIONS = {
    'Ackley'                  : synthetic.Ackley,
    'Beale'                   : synthetic.Beale,
    'Branin'                  : synthetic.Branin,
    'Bukin'                   : synthetic.Bukin,
    'Cosine8'                 : synthetic.Cosine8,
    'DixonPrice'              : synthetic.DixonPrice,
    'DropWave'                : synthetic.DropWave,
    'EggHolder'               : synthetic.EggHolder,
    'Griewank'                : synthetic.Griewank,
    'Hartmann'                : synthetic.Hartmann,
    'HolderTable'             : synthetic.HolderTable,
    'Levy'                    : synthetic.Levy,
    'Michalewicz'             : synthetic.Michalewicz,
    'Powell'                  : synthetic.Powell,
    'PressureVessel'          : synthetic.PressureVessel,
    'Rastrigin'               : synthetic.Rastrigin,
    'Rosenbrock'              : synthetic.Rosenbrock,
    'Shekel'                  : synthetic.Shekel,
    'SixHumpCamel'            : synthetic.SixHumpCamel,
    'SpeedReducer'            : synthetic.SpeedReducer,
    'StyblinskiTang'          : synthetic.StyblinskiTang,
    'TensionCompressionString': synthetic.TensionCompressionString,
    'ThreeHumpCamel'          : synthetic.ThreeHumpCamel,
    'WeldedBeamSO'            : synthetic.WeldedBeamSO,
}

SYNTHETIC_1D_FUNCTIONS = {
    'Ackley'                  : synthetic.Ackley,
    'DixonPrice'              : synthetic.DixonPrice,
    'Griewank'                : synthetic.Griewank,
    'Levy'                    : synthetic.Levy,
    'Michalewicz'             : synthetic.Michalewicz,
    'Rastrigin'               : synthetic.Rastrigin,
    'StyblinskiTang'          : synthetic.StyblinskiTang,
}

MO_SYNTHETIC_FUNCTIONS = {
    'BraninCurrin' : multi_objective.BraninCurrin,
    'CarSideImpact': multi_objective.CarSideImpact,
    'DH1'          : multi_objective.DH1,
    'DH2'          : multi_objective.DH2,
    'DH3'          : multi_objective.DH3,
    'DH4'          : multi_objective.DH4,
    'DTLZ1'        : multi_objective.DTLZ1,
    'DTLZ2'        : multi_objective.DTLZ2,
    'DTLZ3'        : multi_objective.DTLZ3,
    'DTLZ4'        : multi_objective.DTLZ4,
    'DTLZ5'        : multi_objective.DTLZ5,
    'DTLZ7'        : multi_objective.DTLZ7,
    'GMM'          : multi_objective.GMM,
    'Penicillin'   : multi_objective.Penicillin,
    'ToyRobust'    : multi_objective.ToyRobust,
    'VehicleSafety': multi_objective.VehicleSafety,
    'ZDT1'         : multi_objective.ZDT1,
    'ZDT2'         : multi_objective.ZDT2,
    'ZDT3'         : multi_objective.ZDT3,
}

def truth_at(
    truth   : SyntheticTestFunction,
    X       : np.ndarray | float
) -> np.ndarray:
    """Noiseless values of the truth at the (n, d) actions X."""
    X = np.asarray(X)
    with torch.no_grad():
        return truth(torch.as_tensor(X, dtype=DTYPE), noise=False).numpy()


def construct_function(
    func              : type[SyntheticTestFunction] | type[MultiObjectiveTestProblem],
    dim               : int,
    box               : float,
    num_objectives    : int | None = None,
    seed              : int = 0
):
    """One non-constant instance of func at the requested dim, or None if it has
    no such instance.

    Defaults to the [-box, box] action box and falls back to the function's own
    default bounds. No multi-objective problem accepts a `bounds` argument at
    all, so for those the box is inert and `truth.bounds` is the domain.

    Args:
        func: a `SyntheticTestFunction` or `MultiObjectiveTestProblem` subclass.
        dim: the action dimension wanted.
        box: half-width of the preferred action box.
        num_objectives: m, for the families that take it (DTLZ*, ZDT*, GMM).
        seed: seeds the non-constant probe.
    """
    # TODO: get rid of this function eventually...
    # each candidate is filtered against the constructor's own signature, since
    # the two families take different arguments and neither takes the other's
    takes  = signature(func).parameters
    shared = {'num_objectives': num_objectives} if num_objectives is not None else {}

    tried = []
    for kwargs in ({'dim': dim, 'bounds': [(-box, box)] * dim},
                   {'bounds': [(-box, box)] * dim},
                   {'dim': dim},
                   {}):
        kwargs = {k: v for k, v in (kwargs | shared).items() if k in takes}
        if kwargs in tried:
            continue     # the filter collapsed it onto a candidate already tried
        tried.append(kwargs)

        try:
            truth = func(**kwargs)
        except (TypeError, ValueError, AssertionError, BotorchError):
            continue     # wrong dim, or a box holding no known optimizer

        if truth.dim != dim:
            continue

        # Powell sums over range(dim // 4) and Rosenbrock over range(dim - 1),
        # so below dim 4 and dim 2 they are identically zero. Per objective
        # rather than pooled: a wide objective would otherwise cover a
        # constant one, whose ptp is a column of an (n, m) result.
        probe = sample_actions(bounds=truth.bounds, n=PROBE_SAMPLES,
                               kind='sobol', seed=seed)
        if np.min(np.ptp(truth_at(truth, probe), axis=0)) > 0:
            return truth

    return None

class SyntheticOracle:
    """A synthetic test function plus an observation noise stated as a fraction
    of the function's own spread.

    Function spread is measured once, over a Sobol scan of the whole box.

    Args:
        truth: a `SyntheticTestFunction` *instance*.
        rel_noise_std: noise standard deviation, as a fraction of the spread.
        measure: 'std' for the scan's standard deviation, 'range' for its
            peak-to-peak range.
        n_spread: points in the scan.
        seed: seeds both the scan and the noise draws.

    Attributes:
        sample_max, sample_min, ptp: the scan's largest and smallest value,
            their arguments (`sample_argmax`, `sample_argmin`), and their gap.
        measure_spread: the measured spread, in the function's own units.
        noise_std: the absolute noise standard deviation applied.
    """

    def __init__(
        self,
        truth           : SyntheticTestFunction,
        rel_noise_std   : float = 0.0,
        measure         : str = 'range',
        n_spread        : int = SPREAD_SAMPLES,
        seed            : int = 0
    ):
        if measure not in ('std', 'range'):
            raise ValueError(f"measure must be 'std' or 'range', got {measure!r}")

        self.truth         = truth
        self.rel_noise_std = rel_noise_std
        self.rng           = np.random.default_rng(seed)

        X = sample_actions(
            bounds    = truth.bounds,
            n         = n_spread,
            kind      = 'sobol',
            seed      = seed
        )
        y = truth_at(
            truth = truth,
            X = X
        )
        max_action_idx        = np.argmax(y)
        min_action_idx        = np.argmin(y)
        self.sample_min      = y[min_action_idx]
        self.sample_max      = y[max_action_idx]
        self.sample_argmin    = X[min_action_idx]
        self.sample_argmax    = X[max_action_idx]
        self.ptp              = self.sample_max - self.sample_min
        self.measure_spread   = y.std() if measure == 'std' else self.ptp
        self.noise_std        = rel_noise_std * self.measure_spread

    def __call__(
        self,
        X       : np.ndarray | float,
        noise   : bool = True
    ):
        """Values at the (n, d) actions X, returned (n,).

        The noise is drawn from the instance's own generator, so repeated calls
        advance one stream rather than repeating a seeded draw.
        """
        y = truth_at(self.truth, X)
        if not noise:
            return y

        return y + self.noise_std * self.rng.standard_normal(y.shape)
    
    @classmethod
    def from_name(
        cls,
        func            : str,
        dim             : int,
        box             : float,
        seed            : int = 0,
        rel_noise_std   : float = 0.0,
        measure         : str = 'range',
        n_spread        : int = SPREAD_SAMPLES
    ):
        """One oracle, from arguments plain enough to store and replay."""
        truth = construct_function(
            func    = SYNTHETIC_FUNCTIONS[func],
            dim     = dim,
            box     = box,
            seed    = seed
        )
        if truth is None:
            raise ValueError(f'{func} has no non-constant instance at dim {dim}')

        return cls(
            truth           = truth,
            rel_noise_std   = rel_noise_std,
            measure         = measure,
            n_spread        = n_spread,
            seed            = seed
        )
    
class MO2SO:
    """One scalarization of a multi-objective truth: `truth(X) @ tradeoff`.

    Exposes the `bounds`, `dim` and `noise` keyword a `SyntheticTestFunction`
    instance does, which is the whole interface `truth_at` and `SyntheticOracle`
    ask of a truth.

    Args:
        truth: a `MultiObjectiveTestProblem` *instance*.
        tradeoff: (m,) weights over its objectives.
    """

    def __init__(
        self,
        truth       : MultiObjectiveTestProblem,
        tradeoff    : np.ndarray
    ):
        self.truth    = truth
        self.tradeoff = torch.as_tensor(tradeoff, dtype=DTYPE)
        self.bounds   = truth.bounds
        self.dim      = truth.dim

    def __call__(self, X, noise=True):
        """Values at the (n, d) actions X, returned (n,)."""
        return self.truth(X, noise=noise) @ self.tradeoff


class MOSyntheticOracle:
    """A multi-objective test function as m independent `SyntheticOracle`s.

    Args:
        truth: a `MultiObjectiveTestProblem` *instance*.
        rel_noise_std: noise standard deviation, as a fraction of each
            objective's own spread.
        measure: 'std' or 'range', as `SyntheticOracle` takes it.
        n_spread: points in each spread scan.
        seed: seeds the scans and the noise draws.
        ref_point: (m,) the worst value per objective that still counts, in the
            truth's own units. A `MultiObjectiveTestProblem` supplies its own,
            so it is required only for a list of functions.

    Attributes:
        objectives: the m `SyntheticOracle`s, in the truth's objective order.
        measure_spread: (m,) each objective's measured spread, in its own units.
        noise_std: (m,) the absolute noise standard deviation applied to each.
        ref_point: (m,) the hypervolume reference, in the truth's own units.
        sampled_max_hypervolume: the hypervolume of the scan's own front, the
            multi-objective analog of a `SyntheticOracle`'s `sample_min`.
    """

    def __init__(
        self,
        truth           : MultiObjectiveTestProblem | list[SyntheticTestFunction],
        rel_noise_std   : float = 0.0,
        measure         : str   = 'range',
        n_spread        : int   = SPREAD_SAMPLES,
        seed            : int   = 0,
        ref_point       : np.ndarray | None = None
    ):
        self.truth         = truth
        self.rel_noise_std = rel_noise_std

        if isinstance(truth, MultiObjectiveTestProblem):
            self.objectives = [
                SyntheticOracle(
                    truth           = MO2SO(truth, w),
                    rel_noise_std   = rel_noise_std,
                    measure         = measure,
                    n_spread        = n_spread,
                    seed            = seed + i
                )
                for i, w in enumerate(np.eye(truth.num_objectives))
            ]
        else:
            self.objectives = [
                SyntheticOracle(
                    truth           = synfunc,
                    rel_noise_std   = rel_noise_std,
                    measure         = measure,
                    n_spread        = n_spread,
                    seed            = seed + i
                )
                for i, synfunc in enumerate(truth)  
            ]
        self.measure_spread = np.array([o.measure_spread for o in self.objectives])
        self.noise_std      = np.array([o.noise_std for o in self.objectives])

        if ref_point is None:
            if not isinstance(truth, MultiObjectiveTestProblem):
                raise ValueError('ref_point is required for a list of functions')
            ref_point = truth.ref_point.numpy()
        
        self.ref_point = np.broadcast_to(
            array = np.asarray(
                ref_point, 
                dtype=float
            ),
            shape = (len(self.objectives),)
        )

        X = sample_actions(
            bounds=self.objectives[0].truth.bounds, 
            n=n_spread,
            kind='sobol', 
            seed=seed
        )
        # TODO: make this more efficient in the future. Currently samples each objective many times (see how __call__ works for each objective)
        y = self(X, noise=False)

        self.sampled_max_hypervolume = hypervolume_from_nondominated(
            y[get_nondominated(-y)] - self.ref_point
        )

    def objective(self, index) -> SyntheticOracle:
        """Objective `index`, as a scalar `SyntheticOracle`."""
        return self.objectives[index]

    def __len__(self):
        return len(self.objectives)

    def __getitem__(self, index):
        return self.objectives[index]

    def __call__(self, X, noise=True):
        """Values at the (n, d) actions X, returned (n, m).

        Each column draws from its own objective's generator, so the noise is
        independent across the objectives.
        """
        return np.stack([o(X, noise=noise) for o in self.objectives], axis=-1)
    
    @classmethod
    def from_name(
        cls,
        func            : str,
        dim             : int,
        box             : float,
        seed            : int           = 0,
        rel_noise_std   : float         = 0.0,
        num_objectives  : int | None    = None,
        measure         : str           = 'range',
        n_spread        : int           = SPREAD_SAMPLES
    ):
        """One oracle, from arguments plain enough to store and replay."""
        truth = construct_function(
            func            = MO_SYNTHETIC_FUNCTIONS[func],
            dim             = dim,
            box             = box,
            num_objectives  = num_objectives,
            seed            = seed
        )
        if truth is None:
            raise ValueError(f'{func} has no non-constant instance at dim {dim}')

        return cls(
            truth           = truth,
            rel_noise_std   = rel_noise_std,
            measure         = measure,
            n_spread        = n_spread,
            seed            = seed
        )

@dataclass(frozen=True)
class SyntheticOracleParams:
    """The arguments one groundtruth is built from, plain enough to store.

    Attributes:
        func: a key of either registry.
        objectives: one objective name per output column, in column order.
        dim, box, seed, rel_noise_std, measure, n_spread: as the oracles'
            `from_name` takes them.
        num_objectives: m, for the multi-objective families that take it
            (DTLZ*, ZDT*, GMM). Single-objective params leave it None.
    """

    func            : str
    objectives      : tuple[str, ...]
    dim             : int
    box             : float
    seed            : int           = 0
    rel_noise_std   : float         = 0.0
    measure         : str           = 'range'
    n_spread        : int           = SPREAD_SAMPLES
    num_objectives  : int | None    = None

    def __post_init__(self):
        # json reads a tuple back as a list, and a bare name is one column
        names = ((self.objectives,) if isinstance(self.objectives, str)
                 else tuple(self.objectives))
        object.__setattr__(self, 'objectives', names)

        if self.func not in SYNTHETIC_FUNCTIONS and self.func not in MO_SYNTHETIC_FUNCTIONS:
            raise ValueError(f'{self.func!r} is in neither synthetic registry')
        if not self.multi_objective:
            if self.num_objectives is not None:
                raise ValueError(f'{self.func} is single-objective, so '
                                 'num_objectives does not apply')
            if len(names) != 1:
                raise ValueError(f'{self.func} has one output column, got '
                                 f'{len(names)} objective names')

    @property
    def multi_objective(self):
        """Whether these build a `MOSyntheticOracle`."""
        return self.func in MO_SYNTHETIC_FUNCTIONS

    def build(self) -> SyntheticOracle | MOSyntheticOracle:
        """The oracle itself.

        Its noise stream starts from the seed rather than resuming wherever an
        earlier build left off, so a replay repeats the draws rather than
        continuing them.
        """
        shared = {
            'func'          : self.func,
            'dim'           : self.dim,
            'box'           : self.box,
            'seed'          : self.seed,
            'rel_noise_std' : self.rel_noise_std,
            'measure'       : self.measure,
            'n_spread'      : self.n_spread,
        }
        if not self.multi_objective:
            return SyntheticOracle.from_name(**shared)

        oracle = MOSyntheticOracle.from_name(num_objectives=self.num_objectives,
                                             **shared)
        if len(self.objectives) != len(oracle):
            raise ValueError(f'{self.func} has {len(oracle)} output columns, got '
                             f'{len(self.objectives)} objective names')

        return oracle
