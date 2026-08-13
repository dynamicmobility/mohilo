import numpy as np
import matplotlib.pyplot as plt

def plot_gp_1d(
        ax              : plt.Axes, 
        mu              : np.ndarray,
        std             : np.ndarray,
        action_space    : np.ndarray,
        feedback_idxs   : np.ndarray,
        feedback_values : np.ndarray,
        ground_truth    = None,
        num_std         = 1.0
    ):
    """Plot a 1D GP posterior against the data it was fit to.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        mu: length-``N`` posterior mean over ``action_space``.
        std: length-``N`` posterior standard deviation over ``action_space``.
        action_space: ``(N, 1)`` array of actions the posterior is evaluated on.
        feedback_idxs: indices into ``action_space`` where feedback was given.
        feedback_values: the feedback values at those indices.
        ground_truth: optional callable mapping the action space ``(N, d)`` to a
            length-``N`` array of rewards, drawn as a reference curve.
        num_std: width of the shaded uncertainty band, in standard deviations.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    if action_space.shape[1] != 1:
        raise ValueError(
            f"plot_gp_1d only supports 1D action spaces, got shape {action_space.shape}"
        )

    xs = action_space.ravel()

    # Ground truth reference curve (optional)
    if ground_truth is not None:
        ax.plot(
            xs, ground_truth(action_space), label='Ground Truth', color='blue'
        )

    # Observed feedback points
    idx = feedback_idxs.astype(int)
    ax.scatter(action_space[idx].ravel(), feedback_values, color='red', label='Feedback Data')

    # GP posterior mean and uncertainty band
    ax.plot(xs, mu, label='GP Mean', color='green')
    ax.fill_between(
        xs, mu - num_std * std, mu + num_std * std,
        color='green', alpha=0.2, label=f'GP ±{num_std:g}σ'
    )

    ax.set_xlabel('Action')
    ax.set_ylabel('Reward')
    ax.legend()
    return ax