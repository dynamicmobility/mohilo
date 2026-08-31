from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

# Okabe-Ito, a published colorblind-safe palette
MODEL_COLOR  = '#D55E00'
TRUTH_COLOR  = '#000000'
SAMPLE_COLOR = '#009E73'

# cycled over `overlays`, in the order they are given
OVERLAY_COLORS = ('#0072B2', '#CC79A7', '#E69F00', '#56B4E9')

# (color, linestyle) cycled over `vlines`, in the order they are given
VLINE_STYLES = ((MODEL_COLOR, '--'), (TRUTH_COLOR, ':'))

INK, MUTED = "#0b0b0b", "#52514e"   # chart ink and recessive furniture

def dress_axis(ax: plt.Axes) -> plt.Axes:
    """Apply the house style: recessive grid, muted ticks, no top/right spines."""
    ax.grid(True, color=INK, alpha=0.12, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax

def plot_test_function(
        ax    : plt.Axes,
        X     : np.ndarray,
        y     : np.ndarray,
        title : str = None
    ):
    """Plot a scalar function sampled at scattered points.

    The scan need not be a grid: 1D is sorted before it is drawn and 2D is
    contoured over its own Delaunay triangulation, so a Sobol sequence plots
    correctly either way.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        X: ``(N, d)`` actions the function was evaluated at, ``d`` 1 or 2.
        y: length-``N`` values at them.
        title: optional axes title.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    y = np.asarray(y).ravel()
    if X.shape[1] == 1:
        order = np.argsort(X.ravel())
        ax.plot(X.ravel()[order], y[order], lw=0.8)
    elif X.shape[1] == 2:
        ax.tricontourf(X[:, 0], X[:, 1], y, levels=32)
    else:
        raise ValueError(f'Cannot plot a {X.shape[1]}D function. Got X.shape = {X.shape}')

    if title is not None:
        ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)

    return ax


def plot_fit_1d(
        ax       : plt.Axes,
        x        : np.ndarray,
        mu       : np.ndarray,
        std      : np.ndarray,
        xdata    : np.ndarray,
        ydata    : np.ndarray,
        truth    : np.ndarray      = None,
        paths    : np.ndarray      = None,
        vlines   : dict[str, float] = None,
        band_std : float           = 2.0,
        title    : str             = None
    ) -> plt.Axes:
    """Plot a 1D model fit: a posterior mean and band, optional sample paths and
    truth curve, and the measurements the fit was made from.

    Everything arrives already evaluated and in one shared set of units, so the
    band is whatever ``std`` is: a *latent* posterior standard deviation makes
    it the model's uncertainty about the noiseless function -- what a truth
    curve should fall inside -- rather than a predictive interval for a new
    measurement, which would be wider by the observation noise.

    ``x`` need not be sorted; it is ordered here and every array sharing its
    axis is reordered with it.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        x: ``(n,)`` actions the model was evaluated at.
        mu: ``(n,)`` posterior mean at them.
        std: ``(n,)`` posterior standard deviation at them.
        xdata: ``(N,)`` measured actions.
        ydata: ``(N,)`` measured values.
        truth: optional ``(n,)`` noiseless truth at ``x``.
        paths: optional ``(S, n)`` sample paths from the posterior over ``x``.
        vlines: optional label -> action, drawn as vertical lines styled by
            ``VLINE_STYLES`` in the order given.
        band_std: half-width of the shaded band, in standard deviations.
        title: optional axes title.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    order       = np.argsort(np.ravel(x))
    x, mu, std  = (np.ravel(a)[order] for a in (x, mu, std))
    if truth is not None:
        truth = np.ravel(truth)[order]
    if paths is not None:
        paths = np.atleast_2d(paths)[:, order]
        for i, path in enumerate(paths):
            ax.plot(x, path, color=SAMPLE_COLOR, lw=0.7, alpha=0.35, zorder=1,
                    label=f'{len(paths)} posterior samples' if i == 0 else None)

    ax.fill_between(x, mu - band_std * std, mu + band_std * std,
                    color=MODEL_COLOR, alpha=0.18, zorder=2,
                    label=f'posterior mean $\\pm$ {band_std:g}$\\sigma$')
    ax.plot(x, mu, color=MODEL_COLOR, lw=1.8, ls='--', zorder=3,
            label='posterior mean')
    if truth is not None:
        ax.plot(x, truth, color=TRUTH_COLOR, lw=1.8, zorder=4, label='truth')

    for i, (label, action) in enumerate((vlines or {}).items()):
        color, ls = VLINE_STYLES[i % len(VLINE_STYLES)]
        ax.axvline(action, color=color, ls=ls, lw=1.2, zorder=5, label=label)

    ax.scatter(np.ravel(xdata), np.ravel(ydata), s=8, color=TRUTH_COLOR,
               zorder=-1, label='measurements')

    ax.set_xlabel('action')
    ax.set_ylabel('objective')
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    return ax


def plot_loo_curve(
        ax       : plt.Axes,
        sizes    : np.ndarray,
        scores   : np.ndarray,
        baseline : float = 0.0,
        ylabel   : str = None,
        title    : str = None
    ):
    """Plot a held-out score against dataset size.

    Summarized by median and interquartile band rather than mean and standard
    deviation, because a score like ``R^2`` is heavy-tailed at small sizes and a
    few collapsed subsets would drag the mean well below the typical one.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        sizes: length-``S`` subset sizes, as ``loo_curve`` returns them.
        scores: ``(S, repeats)`` scores at each size.
        baseline: a reference line, ``0.0`` being where ``R^2`` stops beating
            the training mean. None draws none.
        ylabel: optional y label, e.g. ``'$R^2$'``.
        title: optional axes title.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    lo, med, hi = np.percentile(scores, [25, 50, 75], axis=1)

    if baseline is not None:
        ax.axhline(baseline, color=TRUTH_COLOR, ls=':', lw=0.8)
    ax.fill_between(sizes, lo, hi, color=MODEL_COLOR, alpha=0.2, lw=0)
    ax.plot(sizes, med, color=MODEL_COLOR, marker='o', ms=3, lw=1.2)

    ax.set_xlabel('subset size (each fold fit on one fewer)')
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax


def plot_mo_space(
        ax          : plt.Axes,
        true_front  : np.ndarray = None,
        measured    : np.ndarray = None,
        predicted   : np.ndarray = None,
        overlays    : dict = None,
        names       : tuple = None,
        title       : str = None
    ):
    """Plot one 2D objective space: the truth's Pareto front as a line, the
    points a run measured and the front a model infers as clouds, and any number
    of labeled overlays on top.

    Everything arrives already evaluated, and in one common set of units, for
    the same reason ``plot_test_function`` takes ``y``: the caller owns the
    groundtruth and the model, so this module stays on plain arrays.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        true_front: ``(k, 2)`` the truth's own front, sorted along a column so
            it draws as a line.
        measured: ``(N, 2)`` values at the actions a run has measured.
        predicted: ``(k, 2)`` values at the model's inferred Pareto set.
        overlays: label -> ``(n, 2)`` points, drawn in ``OVERLAY_COLORS`` order.
        names: the two axis labels, verbatim.
        title: optional axes title.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    if true_front:
        if true_front.shape[1] != 2:
            raise ValueError(f'the objective space is drawn flat, got '
                            f'{true_front.shape[1]}D')

        ax.plot(true_front[:, 0], true_front[:, 1], color=TRUTH_COLOR, lw=1.5,
                zorder=3, label='true front')
    if measured is not None:
        ax.scatter(*measured.T, s=22, marker='x', lw=1.0, color=SAMPLE_COLOR,
                   alpha=0.7, zorder=2, label='measured')
    if predicted is not None:
        ax.scatter(*predicted.T, s=20, color=MODEL_COLOR, alpha=0.8, zorder=4,
                   label='inferred front')

    for (label, points), color in zip((overlays or {}).items(), cycle(OVERLAY_COLORS)):
        ax.scatter(*np.atleast_2d(points).T, s=90, color=color, zorder=5,
                   edgecolor=TRUTH_COLOR, lw=0.8, label=label)

    if names is not None:
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    return dress_axis(ax)

