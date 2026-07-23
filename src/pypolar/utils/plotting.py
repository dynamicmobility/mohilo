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
    """Plot a 1D GP estimate against a regression dataset.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        gp: a fitted ``GPModel`` (uses ``gp.mu`` and ``gp.std()``).
        regression: a ``Regression`` holding the action space and feedback data.
        ground_truth: optional callable mapping the action space ``(N, d)`` to a
            length-``N`` array of rewards, drawn as a reference curve.
        num_std: width of the shaded uncertainty band, in standard deviations.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    # x = regression.action_space
    if action_space.shape[1] != 1:
        raise ValueError(
            f"plot_gp_1d only supports 1D action spaces, got shape {x.shape}"
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


def plot_pareto_2d(
    ax              : plt.Axes, 
    optimizer,
    regression,
    mo_ground_truth,
):
    x = regression.action_space

    ax.plot(*mo_ground_truth(x).T, label='Ground Truth Pareto Front', color='grey')

    ax.scatter(
        regression.feedback_data[:, 1],
        regression.feedback_data[:, 2],
        color='red', 
        label='Feedback Data',
        s=40,
        zorder=3
    )

    ax.plot(
        optimizer.gps[0].mu, 
        optimizer.gps[1].mu, 
        label='GP Mean Pareto', 
        zorder=2,
        color='blue',
    )
    ax.legend()
    
    return ax