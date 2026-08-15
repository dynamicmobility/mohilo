"""One acquisition function, the box it is maximized over, and the optimizer
settings that maximize it."""

from inspect import signature

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction as BoTorchAcqf
from botorch.optim import optimize_acqf

from pypolar.optimization.gp import DTYPE, NUM_RESTARTS, RAW_SAMPLES, BoTorchGP
from pypolar.optimization.objectives import Objective
from botorch.acquisition import (
    AcquisitionFunction
)

class UniformRandom(AcquisitionFunction):
    def __init__(self):
        raise NotImplementedError()

class AcquisitionFunction:
    """A BoTorch acquisition bound to a search box rather than to a model.

    The search runs in the GP's normalized frame, since that is the frame the
    model was trained in, and the box is mapped into it at every query.
    """

    def __init__(self, acqf: type[BoTorchAcqf], objective: Objective,
                 num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES, bounds=None):
        """
        Args:
            acqf: a BoTorch acquisition class, or a partial of one. Called as
                `acqf(model, **incumbent)`.
            objective: the objective the box defaults to, and whose normalized
                frame the search runs in.
            num_restarts: L-BFGS-B starting points per optimization.
            raw_samples: Sobol samples scanned to pick them.
            bounds: (low, high) actions in raw units, scalar or one per action
                dimension. Defaults to the objective's own `action_bounds`.
        """
        if bounds is None:
            bounds = objective.action_bounds
        if bounds is None:
            raise ValueError(
                f'{objective.name}: without action_bounds the normalized frame is '
                'the box of the points measured so far, which moves as they '
                'arrive; pin action_bounds or pass bounds'
            )

        self.acqf         = acqf
        self.action_box   = bounds
        self.num_restarts = num_restarts
        self.raw_samples  = raw_samples

    def _box(self, objective):
        """The search box in raw units, (2, d), broadcast so a scalar bound
        covers every action dimension."""
        d = objective.xdata.shape[1]
        return np.broadcast_to(
            np.asarray(self.action_box, dtype=float).reshape(2, -1), (2, d)
        )

    def _incumbent(self, objective):
        """The measurement-dependent arguments `self.acqf` takes, from the
        objective's current data. Names a partial already bound are left to it."""
        X = torch.as_tensor(objective.normalized_x, dtype=DTYPE)
        available = {
            'best_f'     : objective.standard_y.max(),
            'X_observed' : X,
            'X_baseline' : X
        }

        taken = signature(self.acqf).parameters
        bound = getattr(self.acqf, 'keywords', {})
        return {name: value for name, value in available.items()
                if name in taken and name not in bound}

    def query(self, model: BoTorchGP, q=1, raw=True):
        """The q actions jointly maximizing the acquisition over the box.

        Args:
            model: the GP the acquisition reads, and whose measurements supply
                its incumbent.
            q: actions returned per call. Only the q-acquisitions take q > 1.
            raw: report in the objective's own units rather than the normalized
                frame.

        Returns:
            (q, d) actions.
        """
        objective = model.objective
        box       = self._box(objective)

        candidate, _ = optimize_acqf(
            acq_function = self.acqf(model.model, **self._incumbent(objective)),
            bounds       = torch.as_tensor(objective.xtransform(box), dtype=DTYPE),
            q            = q,
            num_restarts = self.num_restarts,
            raw_samples  = self.raw_samples
        )

        action = candidate.detach().numpy()
        if not raw:
            return action

        # optimize_acqf is box-constrained, so the round trip through the
        # transform leaves the box only by float round-off
        return np.clip(objective.xtransform.inv(action), box[0], box[1])
