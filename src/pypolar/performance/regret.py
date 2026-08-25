"""Single-objective metric against a known groundtruth: the inference regret of
a recommended action, in spreads of the groundtruth's own range."""

import numpy as np

from pypolar.feedback.synthetic import SyntheticOracle


def normalized_inference_regret(
    raw_recommended_action    : np.ndarray,
    ground_truth              : SyntheticOracle,
    maximize                  : bool = False
):
    """The inference regret of a recommended action.

    Args:
        raw_recommended_action: the recommended action, in raw action units.
        ground_truth: the oracle, whose `sample_min`/`sample_max` and `ptp`
            come from its own Sobol scan of the box.
        maximize: True when a larger true value is better.

    Returns:
        The regret, as a non-negative fraction of the truth's range.
    """
    inferred_val      = ground_truth(raw_recommended_action, noise=False)
    optimal_val       = ground_truth.sample_max if maximize else ground_truth.sample_min
    sign              = -1.0 if maximize else 1.0
    scaled_regret     = sign * (inferred_val - optimal_val) / ground_truth.ptp

    return np.ravel(scaled_regret)[0]
