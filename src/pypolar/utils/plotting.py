from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Okabe-Ito, a published colorblind-safe palette
MODEL_COLOR  = '#D55E00'
TRUTH_COLOR  = '#000000'
SAMPLE_COLOR = '#009E73'

# cycled over `overlays`, in the order they are given
OVERLAY_COLORS = ('#0072B2', '#CC79A7', '#E69F00', '#56B4E9')

# (color, linestyle) cycled over `vlines`, in the order they are given
VLINE_STYLES = ((MODEL_COLOR, '--'), (TRUTH_COLOR, ':'))

INK, MUTED = "#0b0b0b", "#52514e"   # chart ink and recessive furniture

TICK_SIZE   = 12        # tick label size, in points
FONT        = "cmr10"   # matplotlib's bundled Computer Modern roman
MATH_FONT   = "cm"      # mathtext's Computer Modern fontset
LABELPAD_3D = 10        # axis label distance from its ticks on a 3D axes, in points
LABEL_SIZE  = 14        # axis label size, in points
TITLE_SIZE  = 16        # axes title size, in points

def dress_axis(
        ax          : plt.Axes,
        tick_size   : float = TICK_SIZE,
        label_size  : float = LABEL_SIZE,
        title_size  : float = TITLE_SIZE,
        font        : str   = FONT,
        math_font   : str   = MATH_FONT,
        labelpad_3d : float = LABELPAD_3D,
        num_xticks  : int   = None,
        num_yticks  : int   = None,
        num_zticks  : int   = None
    ) -> plt.Axes:
    """Apply the house style: recessive grid, muted ticks, no top/right spines,
    and one font on every text the axes holds when called.

    Args:
        ax: a 2D or 3D ``matplotlib.axes.Axes`` to style.
        tick_size: tick label size, in points.
        label_size: axis label size, in points.
        title_size: axes title size, in points.
        font: font family for every text, e.g. ``"cmr10"``.
        math_font: mathtext fontset for the ``$...$`` parts, e.g. ``"cm"``.
        labelpad_3d: axis label padding, in points; 3D axes only.
        num_xticks, num_yticks, num_zticks: the most ticks that axis draws,
            placed at round values. None keeps matplotlib's own choice.
            ``num_zticks`` needs a 3D axes.

    Returns:
        The ``ax`` that was styled, for chaining.
    """
    ax.grid(True, color=INK, alpha=0.12, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    is_3d = ax.name == "3d"
    axes  = (ax.xaxis, ax.yaxis, ax.zaxis) if is_3d else (ax.xaxis, ax.yaxis)
    for which in ("both", "z") if is_3d else ("both",):
        ax.tick_params(axis=which, colors=MUTED, labelsize=tick_size,
                       labelfontfamily=font)
    if is_3d:
        for axis in axes:
            axis.labelpad = labelpad_3d

    for name, num in zip("xyz", (num_xticks, num_yticks, num_zticks)):
        if num is None:
            continue
        if name == "z" and not is_3d:
            raise ValueError("num_zticks needs a 3D axes")
        # nbins counts the gaps between ticks, so n ticks is n - 1 of them
        ax.locator_params(axis=name, nbins=num - 1)

    # cmr10 has no unicode minus, so tick numbers are drawn through mathtext
    for axis in axes:
        if isinstance(axis.get_major_formatter(), ScalarFormatter):
            axis.get_major_formatter().set_useMathText(True)

    legend = ax.get_legend()
    texts  = [ax.title, *ax.texts, *(legend.get_texts() if legend else []),
              *(text for axis in axes
                for text in (axis.label, axis.get_offset_text(),
                             *axis.get_ticklabels()))]
    for text in texts:
        text.set_fontfamily(font)
        text.set_math_fontfamily(math_font)
    for axis in axes:
        axis.label.set_fontsize(label_size)
    ax.title.set_fontsize(title_size)
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
        title    : str             = None,
        measurement_s: int = 8
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
    ax.plot(x, mu, color=MODEL_COLOR, lw=1.8, ls='--', zorder=4,
            label='posterior mean')
    if truth is not None:
        ax.plot(x, truth, color=TRUTH_COLOR, lw=1.8, zorder=3, label='truth')

    for i, (label, action) in enumerate((vlines or {}).items()):
        color, ls = VLINE_STYLES[i % len(VLINE_STYLES)]
        ax.axvline(action, color=color, ls=ls, lw=1.2, zorder=5, label=label)

    ax.scatter(np.ravel(xdata), np.ravel(ydata), s=measurement_s, color=TRUTH_COLOR,
               zorder=8, label='measurements')

    ax.set_xlabel('action')
    ax.set_ylabel('objective')
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)

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
    if true_front is not None:
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



def _decide_color_kwargs(
        colors : str | np.ndarray = None,
        idx    : np.ndarray       = None
    ) -> dict:
    """Route a color specification to the ``scatter`` keyword that will not
    value-map it.

    Args:
        colors: one matplotlib color, an ``(n, 3|4)`` RGB(A) array of one color
            per point, or None to leave the color unset.
        idx: indices of the rows of ``colors`` being drawn, or None for all.

    Returns:
        The keyword to splat into ``ax.scatter``: ``c`` for a per-point array,
        ``color`` for a single color, and nothing for None.
    """
    if colors is None:
        return {}
    if isinstance(colors, np.ndarray) and colors.ndim == 2:
        return {"c": colors if idx is None else colors[idx]}
    return {"color": colors}


def plot_pareto(
        ax                    : plt.Axes,
        pareto                : np.ndarray,
        nd_idx                : np.ndarray       = None,
        colors                : str | np.ndarray = None,
        objective             : list[str]        = None,
        connect               : bool             = False,
        show_dominated        : bool             = True,
        nondominated_alpha    : float            = 1.0,
        dominated_alpha       : float            = 1.0,
        nondominated_s        : int              = 20,
        dominated_s           : int              = 8,
        outline_nondominated  : float            = 0.0,
        label                 : str              = None,
        set_lims              : bool             = True,
        **plot_kwargs
    ) -> plt.Axes:
    """Plot a 2D Pareto front: its non-dominated points over the dominated ones.

    Args:
        ax: a ``matplotlib.axes.Axes`` to draw on.
        pareto: ``(n, 2)`` values of every point, in the units being plotted.
        nd_idx: indices of the non-dominated rows of ``pareto``; every other row
            is drawn as dominated.
        colors: one matplotlib color, or an ``(n, 3|4)`` RGB(A) array of one
            color per row of ``pareto``.
        objective: the two axis labels, in column order.
        connect: whether the non-dominated points are joined by a line, sorted
            along the first objective.
        show_dominated: whether the dominated points are drawn at all.
        nondominated_alpha: opacity of the non-dominated markers.
        dominated_alpha: opacity of the dominated markers.
        nondominated_s: area of a non-dominated marker, in points squared.
        dominated_s: area of a dominated marker, in points squared.
        outline_nondominated: width of the black edge on the non-dominated
            markers, 0 for none.
        label: what the non-dominated set is called in the legend.
        set_lims: whether the axes are zoomed to the non-dominated points.
        **plot_kwargs: forwarded verbatim to both ``ax.scatter`` calls.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    # TODO: this doesn't work for 3d fronts
    num_objs = pareto.shape[1]
    if num_objs != 2:
        raise NotImplementedError('Only 2D paretos are supported for plotting')

    if objective is None: objective = [''] * num_objs
    c = np.asarray(colors) if isinstance(colors, (list, tuple, np.ndarray)) else colors
    d_idx = np.setdiff1d(np.arange(pareto.shape[0]), nd_idx)

    if show_dominated:
        ax.scatter(
            *(pareto[d_idx].T),
            s       = dominated_s,
            alpha   = dominated_alpha,
            **_decide_color_kwargs(c, d_idx),
            **plot_kwargs,
        )

    ax.scatter(
        *(pareto[nd_idx].T),
        alpha         = nondominated_alpha,
        zorder        = 1,
        s             = nondominated_s,
        edgecolors    = 'black',
        linewidths    = outline_nondominated,
        label         = label,
        **_decide_color_kwargs(c, nd_idx),
        **plot_kwargs,
    )
    if connect:
        pts = pareto[nd_idx]
        ax.plot(
            *(pts[np.argsort(pts[:, 0])].T),
            zorder        = 0,
            color         = 'black',
        )

    if set_lims:
        ax.set_xlim((0.95 * np.min(pareto[nd_idx, 0]), 1.05 * np.max(pareto[nd_idx, 0])))
        ax.set_ylim((0.95 * np.min(pareto[nd_idx, 1]), 1.05 * np.max(pareto[nd_idx, 1])))

    ax.set_xlabel(objective[0], fontsize=16)
    ax.set_ylabel(objective[1], fontsize=16)

    return ax


def plot_pareto_actions(
        ax            : plt.Axes,
        nd_pts        : np.ndarray,
        colors        : str | np.ndarray = None,
        action_labels : list[str]        = None,
        bounds        : np.ndarray       = None,
    ) -> plt.Axes:
    """Plot the actions behind a Pareto front as a path through a 3D action space.

    Args:
        ax: a ``matplotlib.axes.Axes`` with a 3D projection to draw on.
        nd_pts: ``(k, 3)`` non-dominated actions, in the order the line joins
            them.
        colors: one matplotlib color, or a ``(k, 3|4)`` RGB(A) array of one
            color per action.
        action_labels: the three axis labels, in action order.

    Returns:
        The ``ax`` that was drawn on, for chaining.
    """
    # TODO: generalize to 2D too
    ax.plot(
        nd_pts[:, 0], 
        nd_pts[:, 1], 
        nd_pts[:, 2], 
        lw        = 2,
        color     = 'black',
        zorder    = 1
    )
    ax.scatter(
        nd_pts[:, 0], 
        nd_pts[:, 1], 
        nd_pts[:, 2], 
        s             = 25,
        edgecolors    = 'black',
        c             = colors,
        linewidths    = 1,
        zorder        = 2,
        depthshade    = False
    )
    for axis, name in zip(('x', 'y', 'z'), action_labels):
        getattr(ax, f'set_{axis}label')(name)

    if bounds is not None:
        bounds = np.asarray(bounds)
        for axis, b in zip(('x', 'y', 'z'), bounds.T):
            getattr(ax, f'set_{axis}lim')(b)
    return ax
