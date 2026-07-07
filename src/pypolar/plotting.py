def plot_gp_1d(ax, gp, regression, ground_truth=None, num_std=1.0):
    """Plot a 1D GP estimate against a regression dataset.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        gp: a fitted ``BasicGP`` (uses ``gp.mu`` and ``gp.std()``).
        regression: a ``Regression`` holding the action space and feedback data.
        ground_truth: optional callable mapping the action space ``(N, d)`` to a
            length-``N`` array of rewards, drawn as a reference curve.
        num_std: width of the shaded uncertainty band, in standard deviations.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    x = regression.action_space
    if x.shape[1] != 1:
        raise ValueError(
            f"plot_gp_1d only supports 1D action spaces, got shape {x.shape}"
        )

    xs = x.ravel()

    # Ground truth reference curve (optional)
    if ground_truth is not None:
        ax.plot(xs, ground_truth(x), label='Ground Truth', color='blue')

    # Observed feedback points
    fb = regression.feedback_data
    if fb.size > 0:
        idx = fb[:, 0].astype(int)
        ax.scatter(x[idx].ravel(), fb[:, 1], color='red', label='Feedback Data')

    # GP posterior mean and uncertainty band
    mu = gp.mu
    std = gp.std()
    ax.plot(xs, mu, label='GP Mean', color='green')
    ax.fill_between(
        xs, mu - num_std * std, mu + num_std * std,
        color='green', alpha=0.2, label=f'GP ±{num_std:g}σ'
    )

    ax.set_xlabel('Action')
    ax.set_ylabel('Reward')
    ax.legend()
    return ax
