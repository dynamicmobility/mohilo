"""BoTorch synthetic test functions as groundtruths: the registry the
experiments draw from, noiseless evaluation, and an observation noise stated as
a fraction of a function's own spread."""

import numpy as np
import torch

from botorch.exceptions.errors import BotorchError
from botorch.test_functions import SyntheticTestFunction, synthetic

from pypolar.optimization.gp import DTYPE
from pypolar.optimization.objectives import sample_actions

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

# the entries with a non-constant 1D instance, as measured by construct_function
# on [-5, 5]. A box that excludes a function's known optimizer drops it, so
# StyblinskiTang (optimizer at -2.904) leaves this set below a half-width of 2.91.
SYNTHETIC_1D_FUNCTIONS = {
    'Ackley'                  : synthetic.Ackley,
    'DixonPrice'              : synthetic.DixonPrice,
    'Griewank'                : synthetic.Griewank,
    'Levy'                    : synthetic.Levy,
    'Michalewicz'             : synthetic.Michalewicz,
    'Rastrigin'               : synthetic.Rastrigin,
    'StyblinskiTang'          : synthetic.StyblinskiTang,
}


def truth_at(truth: SyntheticTestFunction, X):
    """Noiseless values of the truth at the (n, d) actions X."""
    with torch.no_grad():
        return truth(torch.as_tensor(X, dtype=DTYPE), noise=False).numpy()


def construct_function(func, dim, box, seed=0):
    """One non-constant instance of func at the requested dim, or None if it has
    no such instance. Prefers the [-box, box] action box, falls back to the
    function's own default bounds when it rejects that box."""
    for kwargs in ({'dim': dim, 'bounds': [(-box, box)] * dim},
                   {'bounds': [(-box, box)] * dim},
                   {}):
        try:
            truth = func(**kwargs)
        except (TypeError, ValueError, AssertionError, BotorchError):
            continue     # wrong dim, or a box holding no known optimizer

        if truth.dim != dim:
            continue

        # Powell sums over range(dim // 4) and Rosenbrock over range(dim - 1),
        # so below dim 4 and dim 2 they are identically zero
        probe = sample_actions(bounds=truth.bounds, n=PROBE_SAMPLES,
                               kind='sobol', seed=seed)
        if np.ptp(truth_at(truth, probe)) > 0:
            return truth

    return None


class SyntheticFunction:
    """A synthetic test function plus an observation noise stated as a fraction
    of the function's own spread.

    The spread is measured once, over a Sobol scan of the whole box. That is
    what makes `rel_noise_std` mean the same difficulty across functions whose
    ranges differ by orders of magnitude, and it is the only way a sequential
    loop can use the convention at all: it adds one point at a time, and a
    single point has no spread of its own to take a fraction of.

    Args:
        truth: a `SyntheticTestFunction` *instance*.
        rel_noise_std: noise standard deviation, as a fraction of the spread.
        measure: 'std' for the scan's standard deviation, 'range' for its
            peak-to-peak range.
        n_spread: points in the scan.
        seed: seeds both the scan and the noise draws.

    Attributes:
        spread: the measured spread, in the function's own units.
        noise_std: the absolute noise standard deviation applied.
    """

    def __init__(self, truth: SyntheticTestFunction, rel_noise_std=0.0,
                 measure='std', n_spread=SPREAD_SAMPLES, seed=0):
        if measure not in ('std', 'range'):
            raise ValueError(f"measure must be 'std' or 'range', got {measure!r}")

        self.truth         = truth
        self.rel_noise_std = rel_noise_std
        self.rng           = np.random.default_rng(seed)

        y = truth_at(truth, sample_actions(bounds=truth.bounds, n=n_spread,
                                           kind='sobol', seed=seed))
        self.spread    = y.std() if measure == 'std' else np.ptp(y)
        self.noise_std = rel_noise_std * self.spread

    def __call__(self, X, noise=True):
        """Values at the (n, d) actions X, returned (n,).

        The noise is drawn from the instance's own generator, so repeated calls
        advance one stream rather than repeating a seeded draw.
        """
        y = truth_at(self.truth, X)
        if not noise:
            return y

        return y + self.noise_std * self.rng.standard_normal(y.shape)
