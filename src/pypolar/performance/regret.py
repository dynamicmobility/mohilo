"""Single-objective metrics against a known groundtruth: regret in objective
value, and distance in action space to the true optimizer."""

import numpy as np

from botorch.test_functions import SyntheticTestFunction

from pypolar.feedback.synthetic import truth_at
from pypolar.optimization.objectives import Objective


def regret(truth: SyntheticTestFunction, objective: Objective, inferred,
           maximize=False):
    """Simple and inference regret, both against the true optimum, in the
    objective's own units.

    Simple regret is the best *noiseless* value sampled so far, so a lucky draw
    of the noise cannot flatter it. Inference regret is the true value at the
    action the run would recommend right now -- the posterior argmax -- which
    is what a human-in-the-loop study hands back to a subject.

    `maximize` says which direction `truth` is optimized in, which sets both
    what "best sampled" means and which way the gap is subtracted. A regret is
    a distance from the optimum, so it is non-negative either way: minimizing
    scores `value - optimal_value` over the sampled minimum, maximizing scores
    `optimal_value - value` over the sampled maximum.

    It is a separate argument from `objective.maximize` because the two are
    different claims -- the objective's flag orients its own standardization,
    this one describes the truth being scored against -- but in a run where the
    objective *is* this truth they must agree, so pass the same value.

    Args:
        truth: the groundtruth function, whose `optimal_value` is the target.
            For a botorch synthetic built with `negate=False`, that is the
            global minimum, which is the `maximize=False` case.
        objective: every measurement taken so far; only its actions are read,
            since the values are scored noiselessly.
        inferred: the (d,) recommended action.
        maximize: True when a larger true value is better.

    Returns:
        (simple, inference), both non-negative.
    """
    sampled = truth_at(truth, objective.xdata)
    best    = sampled.max() if maximize else sampled.min()
    sign    = -1.0 if maximize else 1.0

    return (
        sign * (best - truth.optimal_value),
        sign * (truth_at(truth, inferred[None]).item() - truth.optimal_value)
    )


def action_distance(truth: SyntheticTestFunction, objective: Objective, inferred):
    """The same two quantities in action space: how far the closest sampled
    action, and the recommended one, sit from the true optimizer.

    Measured in the objective's normalized frame, so each dimension is divided
    by its own span before the norm is taken and no dimension dominates because
    its units happen to be larger. A distance of 0.1 is a tenth of a box span;
    the whole box has diagonal sqrt(d).

    This asks a different question than `regret` does, and the two can disagree:
    a recommendation in a neighbouring basin of a multimodal truth is far here
    and possibly close there, and a flat optimum inverts that. It is the
    quantity a study pays for directly, since the recommendation it hands a
    subject is an action.

    Returns:
        (simple, inference), in box spans.
    """
    # a global optimum is a global optimum wherever it sits, so the distance is
    # to the nearest of them rather than to a designated one
    optimizers = objective.xtransform(truth.optimizers.numpy())

    def nearest(actions):
        gaps = objective.xtransform(actions)[:, None] - optimizers[None]
        return np.linalg.norm(gaps, axis=-1).min(axis=1)

    return nearest(objective.xdata).min(), nearest(inferred[None]).item()
