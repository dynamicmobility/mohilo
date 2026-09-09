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
from pypolar.optimization.objectives import as_bounds, sample_actions
from pypolar.utils.pareto import get_nondominated, hypervolume_from_nondominated, reference_point

SPREAD_SAMPLES = 4096   # Sobol points a spread is measured over
PROBE_SAMPLES  = 32     # Sobol points a candidate instance is probed at

class IdealPoint(SyntheticTestFunction): # TODO: check this with plotting
    r"""A quadratic bowl with a single optimum at a declared point.

        f(x) = sum_k weights_k * (x_k - optimum_k)^2 + offset

    Stated in the minimizing sense, as every BoTorch test function is, so the
    optimum is a minimum and its value is `offset`. Direction is `Objective`'s
    to state, through `maximize`.

    Args:
        optimum: (d,) the minimizer, or a scalar shared by every dimension.
            Defaults to the origin, so the class is constructible from a `dim`
            and a `box` alone, the way a registry entry is.
        weights: per-dimension curvature, a scalar or (d,). Larger is sharper.
        offset: the value at the optimum.
        dim: d. Needed only to widen a scalar `optimum`.
        bounds: (low, high) pairs, one per dimension -- the orientation a
            `SyntheticTestFunction` constructor takes. Defaults to the
            zero-centered box just wide enough to hold the optimum, at half-width
            1 or more, so bowls with different optima share a box by default.
        noise_std, negate, dtype: as `SyntheticTestFunction` takes them.
    """

    def __init__(
        self,
        optimum     : np.ndarray | float   = 0.0,
        weights     : np.ndarray | float   = 1.0,
        offset      : float                = 0.0,
        dim         : int | None           = None,
        bounds      : list[tuple[float, float]] | None = None,
        noise_std   : float | None         = None,
        negate      : bool                 = False,
        dtype       : torch.dtype          = DTYPE
    ):
        optimum = np.atleast_1d(np.asarray(optimum, dtype=float))
        if optimum.ndim != 1:
            raise ValueError('optimum must be a scalar or (d,), got shape '
                             f'{optimum.shape}')
        if dim is not None:
            if optimum.size == 1:
                optimum = np.full(dim, optimum.item())
            elif optimum.size != dim:
                raise ValueError(f'optimum has {optimum.size} entries against '
                                 f'dim {dim}')

        weights = np.asarray(weights, dtype=float)
        if weights.size not in (1, optimum.size):
            raise ValueError(f'weights has {weights.size} entries against '
                             f'{optimum.size} action dimensions')
        weights = np.broadcast_to(weights, optimum.shape).astype(float)

        self.dim             = optimum.size
        self.continuous_inds = list(range(self.dim))
        if bounds is None:
            half   = max(1.0, float(np.abs(optimum).max()))
            bounds = [(-half, half)] * self.dim

        # the base class checks a custom box against these, so a box excluding
        # the optimum raises here rather than silently moving it
        self._optimizers    = [tuple(optimum)]
        self._optimal_value = float(offset)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds,
                         dtype=dtype)

        self.register_buffer('optimum', torch.tensor(optimum, dtype=dtype))
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))
        self.offset = float(offset)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return ((X - self.optimum) ** 2 * self.weights).sum(-1) + self.offset


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
    'IdealPoint'              : IdealPoint,
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
    'IdealPoint'              : IdealPoint,
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


def _bound_pairs(box, dim):
    """A box, as the (low, high)-per-dimension list a botorch constructor takes."""
    return [tuple(pair) for pair in as_bounds(box, dim=dim).T.tolist()]


def construct_function(
    func              : type[SyntheticTestFunction] | type[MultiObjectiveTestProblem],
    dim               : int,
    box               : float,
    num_objectives    : int | None = None,
    seed              : int = 0
):
    """One non-constant instance of func at the requested dim, or None if it has
    no such instance.

    Defaults to the action box `box` states and falls back to the function's own
    default bounds. No multi-objective problem accepts a `bounds` argument at
    all, so for those the box is inert and `truth.bounds` is the domain.

    Args:
        func: a `SyntheticTestFunction` or `MultiObjectiveTestProblem` subclass.
        dim: the action dimension wanted.
        box: the preferred action box, anything `as_bounds` takes -- a scalar
            half-width for `[-box, box]`, or a (low, high) pair. None when the
            function states its own.
        num_objectives: m, for the families that take it (DTLZ*, ZDT*, GMM).
        seed: seeds the non-constant probe.
    """
    # TODO: get rid of this function eventually..it's a bit weird since it tries making a function over and over until it gets it right.

    # each candidate is filtered against the constructor's own signature, since
    # the two families take different arguments and neither takes the other's
    takes  = signature(func).parameters
    shared = {'num_objectives': num_objectives} if num_objectives is not None else {}

    # botorch's constructors take the transposed (low, high)-per-dimension form
    pairs = None if box is None else _bound_pairs(box, dim)
    boxed = [] if box is None else [{'dim': dim, 'bounds': pairs},
                                    {'bounds': pairs}]

    tried = []
    for kwargs in (*boxed, {'dim': dim}, {}):
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

    Attributes:
        oracles: the m `SyntheticOracle`s, in the truth's objective order.
        measure_spread: (m,) each objective's measured spread, in its own units.
        noise_std: (m,) the absolute noise standard deviation applied to each.
        scan_actions, scan_values: the (n_spread, d) Sobol scan of the box and
            the (n_spread, m) noiseless values there.

    The truth carries no direction of its own -- BoTorch states every one of its
    functions in the minimizing sense -- so which way each column is optimized,
    and hence what its hypervolume means, comes from a `DecoupledObjectives`
    handed to `max_hypervolume`.
    """

    def __init__(
        self,
        truth           : MultiObjectiveTestProblem | list[SyntheticTestFunction],
        rel_noise_std   : float = 0.0,
        measure         : str   = 'range',
        n_spread        : int   = SPREAD_SAMPLES,
        seed            : int   = 0,
    ):
        self.truth         = truth
        self.rel_noise_std = rel_noise_std

        if isinstance(truth, MultiObjectiveTestProblem):
            self.oracles = [
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
            self.oracles = [
                SyntheticOracle(
                    truth           = synfunc,
                    rel_noise_std   = rel_noise_std,
                    measure         = measure,
                    n_spread        = n_spread,
                    seed            = seed + i
                )
                for i, synfunc in enumerate(truth)  
            ]
        self.measure_spread = np.array([o.measure_spread for o in self.oracles])
        self.noise_std      = np.array([o.noise_std for o in self.oracles])

        self.scan_actions = sample_actions(
            bounds    = self.oracles[0].truth.bounds,
            n         = n_spread,
            kind      = 'sobol',
            seed      = seed
        )
        # TODO: make this more efficient in the future. Currently samples each objective many times (see how __call__ works for each objective)
        self.scan_values = self(self.scan_actions, noise=False)

        self._max_hypervolume = {}

    def max_hypervolume(self, objectives, ref_point=None):
        """The hypervolume of the scan's own front, in the directions
        `objectives` states -- the best a finite scan of the box manages, and
        the denominator a run's attained hypervolume is a fraction of.

        Args:
            objectives: a `DecoupledObjectives`, one per truth column in the
                truth's own order. Only its directions are read.
            ref_point: (m,) the worst value per objective that still counts, in
                the truth's own units. None reads it off the scan.

        Returns:
            The hypervolume as a float.
        """
        if objectives.num_objectives != len(self.oracles):
            raise ValueError(f'{objectives.num_objectives} objectives '
                             f'{objectives.names} against {len(self.oracles)} '
                             'truth columns; they are positional, so one per '
                             "column in the truth's order")

        if ref_point is None:
            ref_point = reference_point(values   = self.scan_values,
                                        maximize = objectives.maximize)

        # the direction is the whole key: the same scan against the same
        # reference measures a different region read the other way
        key = (tuple(objectives.signs), tuple(np.ravel(ref_point)))
        if key not in self._max_hypervolume:
            values = objectives.maximization_space(self.scan_values)
            ref    = objectives.maximization_space(ref_point)[0]
            self._max_hypervolume[key] = hypervolume_from_nondominated(
                ref - values[get_nondominated(values)]
            )

        return self._max_hypervolume[key]

    def get_oracle(self, index) -> SyntheticOracle:
        """Objective `index`, as a scalar `SyntheticOracle`."""
        return self.oracles[index]
    
    @property
    def bounds(self):
        bs = [self.oracles[i].truth.bounds for i in range(len(self.oracles))]
        bs = np.asarray(bs)
        lo, hi = bs[:, 0].max(0), bs[:, 1].min(0)
        if np.any(lo > hi):
            raise Exception(f'Found incompatible action bounds. collective lows = {lo}, collective highs = {hi}')
        # return bs
        return np.stack([lo, hi])

    def __len__(self):
        return len(self.oracles)

    def __getitem__(self, index):
        return self.oracles[index]

    def __call__(self, X, noise=True):
        """Values at the (n, d) actions X, returned (n, m).

        Each column draws from its own objective's generator, so the noise is
        independent across the objectives.
        """
        return np.stack([o(X, noise=noise) for o in self.oracles], axis=-1)
    
    @classmethod
    def from_name(
        cls,
        func            : str,
        dim             : int           = None,
        box             : float         = None,
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
            `from_name` takes them. `box` is anything `as_bounds` accepts -- a
            scalar half-width, or a (low, high) pair.
        num_objectives: m, for the multi-objective families that take it
            (DTLZ*, ZDT*, GMM). Single-objective params leave it None.
        optima: one ideal point per output column, `IdealPoint` only. Setting it
            is what makes a single-objective function multi-objective: the truth
            becomes m bowls, one per entry, sharing the one box `box` states.
            The shared box is required rather than defaulted because
            `MOSyntheticOracle.bounds` *intersects* its members' boxes, so bowls
            left on their own defaults would land on a box excluding some of
            their own optima.
        ref_point: (m,) hypervolume reference in the truth's own units, or None
            to let a metric read one off the truth's scan. Recorded here rather
            than on the oracle so a metric scores a run against the reference
            the run was optimized against; a hypervolume is only comparable
            under one reference. Multi-objective only.
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
    optima          : tuple | None  = None
    ref_point       : tuple | None  = None

    def __post_init__(self):
        # json reads a tuple back as a list, and a bare name is one column
        names = ((self.objectives,) if isinstance(self.objectives, str)
                 else tuple(self.objectives))
        object.__setattr__(self, 'objectives', names)
        if self.ref_point is not None:
            # TODO: only this way because of dataclass frozen. does it need to be frozen?
            object.__setattr__(self, 'ref_point',
                               tuple(np.ravel(self.ref_point).tolist()))

        # a list field would make the frozen dataclass unhashable, and json
        # reads every sequence back as one
        if self.box is not None and np.ndim(self.box) > 0:
            object.__setattr__(self, 'box',
                               tuple(map(float, np.ravel(self.box))))
        if self.optima is not None:
            object.__setattr__(self, 'optima',
                               tuple(tuple(map(float, np.ravel(o)))
                                     for o in self.optima))

        if self.func not in SYNTHETIC_FUNCTIONS and self.func not in MO_SYNTHETIC_FUNCTIONS:
            raise ValueError(f'{self.func!r} is in neither synthetic registry')

        if self.optima is not None:
            if self.func != 'IdealPoint':
                raise ValueError(f'optima states one ideal point per column, '
                                 f'which only IdealPoint takes, not {self.func}')
            if self.box is None:
                raise ValueError('optima needs a box, since the bowls share one')
            if self.num_objectives is not None:
                raise ValueError('optima already states m, so num_objectives '
                                 'does not apply')
            if len(self.optima) != len(names):
                raise ValueError(f'{len(self.optima)} optima against '
                                 f'{len(names)} objective names; they are one '
                                 'per output column')

        if not self.multi_objective:
            if self.num_objectives is not None:
                raise ValueError(f'{self.func} is single-objective, so '
                                 'num_objectives does not apply')
            if len(names) != 1:
                raise ValueError(f'{self.func} has one output column, got '
                                 f'{len(names)} objective names')
            if self.ref_point is not None:
                raise ValueError(f'{self.func} is single-objective, so '
                                 'ref_point does not apply')

    @property
    def multi_objective(self):
        """Whether these build a `MOSyntheticOracle`."""
        return self.func in MO_SYNTHETIC_FUNCTIONS or self.optima is not None

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

        if self.optima is not None:
            # one bowl per column, every one on the same box, so the
            # intersection `MOSyntheticOracle.bounds` takes is that box itself
            bounds = _bound_pairs(self.box, self.dim)
            oracle = MOSyntheticOracle(
                truth           = [IdealPoint(optimum=o, dim=self.dim,
                                              bounds=bounds)
                                   for o in self.optima],
                rel_noise_std   = self.rel_noise_std,
                measure         = self.measure,
                n_spread        = self.n_spread,
                seed            = self.seed
            )
            return oracle

        # ref_point is recorded, not built in: it states how a hypervolume is
        # read, which is a metric's argument rather than the truth's property
        oracle = MOSyntheticOracle.from_name(num_objectives=self.num_objectives,
                                             **shared)
        if len(self.objectives) != len(oracle):
            raise ValueError(f'{self.func} has {len(oracle)} output columns, got '
                             f'{len(self.objectives)} objective names')

        return oracle
