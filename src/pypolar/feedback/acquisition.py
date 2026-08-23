"""One acquisition function, the box it is maximized over, and the optimizer
settings that maximize it."""

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
from botorch.sampling import SobolQMCNormalSampler


from pypolar.optimization.gp import DTYPE, NUM_RESTARTS, RAW_SAMPLES, BoTorchGP, NoiseModel
from pypolar.optimization.objectives import DecoupledObjectives, Objective

# Acquisition function stuff
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

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
        )
    }
    if strategy not in factories:
        raise ValueError(f'no acquisition for {strategy!r}')

    return factories[strategy]

def acqusition_factory_2d():
    # TODO: implement this when doing MOGP
    pass


class AcquisitionFunction:
    """A BoTorch acquisition bound to a search box rather than to a model.

    The search runs in the GP's normalized frame, since that is the frame the
    model was trained in, and the box is mapped into it at every query.
    """

    def __init__(self, acqf: type[BoTorchAcqf],
                 objective: Objective | DecoupledObjectives,
                 num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES, bounds=None):
        """
        Args:
            acqf: a BoTorch acquisition class, or a partial of one. Called as
                `acqf(model, **incumbent)`.
            objective: the objective, or collection of them, the box defaults
                to and whose action dimension it is broadcast against.
            num_restarts: L-BFGS-B starting points per optimization.
            raw_samples: Sobol samples scanned to pick them.
            bounds: (low, high) actions in raw units, scalar or one per action
                dimension. Defaults to the objective's own `action_bounds`.
        """
        if bounds is None:
            self.bounds = objective.action_bounds
        if self.bounds is None:
            raise ValueError(
                'without action_bounds the normalized frame is the box of the '
                'points measured so far, which moves as they arrive; pin '
                'action_bounds or pass bounds'
            )

        self.acqf         = acqf
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

        return args

    def query(self, model: BoTorchGP, q=1, raw=True):
        """The q actions jointly maximizing the acquisition over the box.

        Args:
            model: the GP the acquisition reads, and whose measurements supply
                its incumbent. A `BoTorchGP` or a `ScalarizedGP`; a
                `DecoupledMOGP` is multi-output and has no scalar to maximize.
            q: actions returned per call. Only the q-acquisitions take q > 1.
            raw: report in the objective's own units rather than the normalized
                frame.

        Returns:
            (q, d) actions.
        """
        box = np.broadcast_to(
            np.asarray(self.bounds, dtype=float).reshape(2, -1), (2, model.objective.action_dim)
        )

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

        # TODO: check this still works with current setup
        # optimize_acqf is box-constrained, so the round trip through the
        # transform leaves the box only by float round-off
        return np.clip(model.frame.inv(action), box[0], box[1])
