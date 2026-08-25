"""One acquisition function, the optimizer settings that maximize it, and an
optional box override for the model's own."""

from inspect import signature

import numpy as np
import torch
from functools import partial
from botorch.acquisition import AcquisitionFunction as BoTorchAcqf
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning


from pypolar.optimization.gp import DTYPE, NUM_RESTARTS, RAW_SAMPLES, BoTorchGP, NoiseModel
from pypolar.optimization.objectives import as_bounds

# Acquisition function stuff
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

# Multi-objective acquisition stuff
REF_POINT       = -2.0  # hypervolume reference per objective, in maximization space
PARTITION_ALPHA = 0.0   # exact box decomposition; botorch's own default is >0 only for m > 4
MO_FANTASIES    = 8     # qhvkg: fantasy models per candidate; cost is linear in it
NUM_PARETO      = 10    # qhvkg: pareto points its inner problem carries

def acquisition_factory_1d(
    strategy        : str,
    seed            : int,
    ucb_beta        : float = UCB_BETA,
    num_fantasies   : int = NUM_FANTASIES,
    mc_samples      : int = MC_SAMPLES,
    prune_baseline  : bool = PRUNE_BASELINE
):
    # TODO: use better guard than is_fitted..
    # if strategy == 'lognei' and NoiseModel.coerce(GP_NOISE).is_fitted:
    #     raise ValueError("'lognei' needs a FixedNoiseGaussianLikelihood, so set "
    #                      'GP_NOISE = plr.NoiseModel.pinned(...) to use it')

    factories = {
        'ucb'    : partial(
            UpperConfidenceBound, 
            beta = ucb_beta
        ),
        'logei'  : LogExpectedImprovement,
        'lognei' : partial(
            LogNoisyExpectedImprovement, 
            num_fantasies = num_fantasies
        ),
        'qlognei': partial(
            qLogNoisyExpectedImprovement,
            sampler        = SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed),
            prune_baseline = prune_baseline
        ),
        'ts'     : PathwiseThompsonSampling
    }
    if strategy not in factories:
        raise ValueError(f'no acquisition for {strategy!r}')

    return factories[strategy]

def acquisition_factory_2d(
    strategy        : str,
    seed            : int,
    num_objectives  : int,
    ref_point       : float = REF_POINT,
    mc_samples      : int   = MC_SAMPLES,
    prune_baseline  : bool  = PRUNE_BASELINE,
    alpha           : float = PARTITION_ALPHA,
    num_fantasies   : int   = MO_FANTASIES,
    num_pareto      : int   = NUM_PARETO
):
    """A multi-objective acquisition, for a `DecoupledMOGP`'s `ModelListGP`.

    Args:
        strategy: 'qlognehvi', 'qlognparego', 'qhvkg' or 'qlogehvi'.
        seed: the QMC sampler's seed.
        num_objectives: m, the length `ref_point` is broadcast to.
        ref_point: hypervolume reference on every objective, in maximization
            space. One scalar covers all m because `feedback()` standardizes.
        mc_samples: QMC samples per acquisition evaluation.
        prune_baseline: drop measured points that cannot be on the front.
        alpha: box-decomposition approximation; 0.0 is exact.
        num_fantasies: qhvkg's fantasy models per candidate.
        num_pareto: qhvkg's pareto points per fantasy.

    Returns:
        A partial called as `acqf(model, **incumbent)`.
    """
    ref     = torch.full((num_objectives,), float(ref_point), dtype=DTYPE)
    sampler = lambda: SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed)

    factories = {
        # scores the improvement over the posterior at X_baseline, so no
        # front is ever built out of the measured values
        'qlognehvi'  : partial(
            qLogNoisyExpectedHypervolumeImprovement,
            ref_point      = ref,
            sampler        = sampler(),
            prune_baseline = prune_baseline,
            alpha          = alpha
        ),
        # a fresh random Chebyshev scalarization per call, then noisy log-EI
        'qlognparego': partial(
            qLogNParEGO,
            sampler        = sampler(),
            prune_baseline = prune_baseline
        ),
        # one-step lookahead on the posterior-mean hypervolume
        'qhvkg'      : partial(
            qHypervolumeKnowledgeGradient,
            ref_point     = ref,
            num_fantasies = num_fantasies,
            num_pareto    = num_pareto
        ),
        # the baseline: takes the current front as known rather than
        # integrating over it, so it is the one to beat under noise
        'qlogehvi'   : partial(qLogExpectedHypervolumeImprovement, ref_point=ref),
    }
    if strategy not in factories:
        raise ValueError(f'no acquisition for {strategy!r}')

    return factories[strategy]


class AcquisitionFunction:
    """A BoTorch acquisition and the optimizer settings that maximize it.

    It carries no objective. The box searched and the frame searched in both
    come from the model handed to `query`, which is the only object that knows
    them consistently: a box stated against one objective and a frame taken
    from another is a search over the wrong region. `bounds` overrides the box.
    """

    def __init__(
        self,
        acqf            : type[BoTorchAcqf],
        num_restarts    : int               = NUM_RESTARTS,
        raw_samples     : int               = RAW_SAMPLES,
        bounds          : np.ndarray | list = None
    ):
        """
        Args:
            acqf: a BoTorch acquisition class, or a partial of one. Called as
                `acqf(model, **incumbent)`.
            num_restarts: L-BFGS-B starting points per optimization.
            raw_samples: Sobol samples scanned to pick them.
            bounds: (2, d) [[low, ...], [high, ...]] actions in raw units, or
                any spelling `as_bounds` takes. Defaults to the queried model's
                `action_bounds`.
        """
        self.acqf         = acqf
        self.bounds       = bounds
        self.num_restarts = num_restarts
        self.raw_samples  = raw_samples

    def _incumbent(self, model):
        """The measurement-dependent arguments `self.acqf` takes, read off the
        model. Names a partial already bound are left to it.

        Built lazily: a scalarized incumbent costs a posterior evaluation over
        every measured action, and no UCB-family acquisition wants one.
        """
        # TODO: get rid of this if possible. looks for args in botorch acquisition function, automatically comes up with best f for logei
        taken  = signature(self.acqf).parameters
        bound  = getattr(self.acqf, 'keywords', {})
        wanted = lambda name: name in taken and name not in bound

        args = {}
        if wanted('posterior_transform') and model.transform is not None:
            args['posterior_transform'] = model.transform
        if wanted('X_observed') or wanted('X_baseline'):
            X = torch.as_tensor(model.measured_x, dtype=DTYPE)
            args.update({name: X for name in ('X_observed', 'X_baseline')
                         if wanted(name)})
        if wanted('best_f'):
            args['best_f'] = model.incumbent()
        if wanted('partitioning'):
            args['partitioning'] = self._partitioning(model, bound.get('ref_point'))

        return args

    @staticmethod
    def _partitioning(model, ref_point):
        """The box decomposition a hypervolume acquisition scores against.

        The front comes from the posterior mean at the measured actions, not
        from the measurements: the objectives are decoupled, so no action
        generally carries all m of them and there is no observed (n, m) to take
        a front of. Where the designs do coincide this is those measurements,
        denoised.
        """
        if ref_point is None:
            raise ValueError('a hypervolume acquisition needs a ref_point; bind '
                             'one with acquisition_factory_2d')

        mu = model.posterior_at(model.measured_x, normalized=True)[0]
        return NondominatedPartitioning(
            ref_point = torch.as_tensor(ref_point, dtype=DTYPE),
            Y         = torch.as_tensor(mu, dtype=DTYPE)
        )

    def query(self, model: BoTorchGP, q=1, raw=True):
        """The q actions jointly maximizing the acquisition over the box.

        Args:
            model: the GP the acquisition reads, whose measurements supply its
                incumbent and whose objective supplies the box. A `BoTorchGP`
                or a `ScalarizedGP` for the single-output acquisitions, a
                `DecoupledMOGP` for the multi-objective ones.
            q: actions returned per call. Only the q-acquisitions take q > 1.
            raw: report in the objective's own units rather than the normalized
                frame.

        Returns:
            (q, d) actions.
        """
        bounds = model.action_bounds if self.bounds is None else self.bounds
        if bounds is None:
            raise ValueError(
                'without action_bounds the normalized frame is the box of the '
                'points measured so far, which moves as they arrive; pin '
                'action_bounds on the objective or pass bounds'
            )

        box = as_bounds(bounds, model.action_dim)

        candidate, _ = optimize_acqf(
            acq_function = self.acqf(model.model, **self._incumbent(model)),
            bounds       = torch.as_tensor(model.frame(box), dtype=DTYPE),
            q            = q,
            num_restarts = self.num_restarts,
            raw_samples  = self.raw_samples
        )

        action = candidate.detach().numpy()
        if not raw:
            return action

        # optimize_acqf is box-constrained, so the round trip through the
        # transform leaves the box only by float round-off
        return np.clip(model.frame.inv(action), box[0], box[1])
