"""One run's inferred Pareto front as it fills in, one column per trial count,
in objective space beside the actions that produced it.

A dataset stores the fits themselves, so each column is the posterior that trial
actually held, not a refit. Every column is drawn in one axis box, shared across
the figure, so the front's movement between trials is the only thing that moves.

`--trials` counts trials, not records. A dataset records each trial *after* its
own action is measured, so its index `i` holds the fit to `i + 1` trials; column
`n` is index `n - 1`, and column 0 is the prior, which no record holds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

import pypolar as plr

DATASET     = Path('hilo/output/experiments/Aug28_Neil/MT01.json')
OUTPUT_DIR  = Path('hilo/output/')

SCAN       = 2**14      # Sobol points the front is read off
SEED       = 95
DPI        = 300

ACTION_LABELS = ['Hip Flex. Scale', 'Hip Ext. Scale', 'Delay']   # action dimensions, in order
SLICES        = 3       # delay levels the colormap key is drawn at
SLICE_WIDTH   = 0.6     # key column width, relative to one trial column
SLICE_GAP     = 1.0     # inches between stacked slices: room for one's x label and the next one's title
PRIOR_CLOUD   = 256     # scan points the prior's action panel draws


def front_order(mu, raw_mu):
    """Sorted nondominated indexes of the Pareto front."""
    nd_idx = plr.get_nondominated_tol(mu)

    return nd_idx[np.argsort(raw_mu[nd_idx, 0])]


def trial_scan(dataset: plr.ExperimentDataset, trial: int, scan: int, seed: int):
    """One column's content: the objectives the fit saw, the `(n, d)` scan
    actions, the `(n, m)` posterior mean there in raw units, and the front's
    sorted indexes.

    Args:
        dataset: the run to read.
        trial: how many trials the fit was given. A record is written *after*
            its own action is measured, so this is dataset index `trial - 1`.
            0 is the prior. A completed run's last record repeats the last
            trial's fit, so the largest distinct column is `len(dataset) - 1`.

    The prior is written out rather than read, because no record holds it and
    none can: an objective with no measurements builds no y transform, so no GP
    can be made from one. `ZeroMean` over no data leaves the posterior at the
    prior, whose mean is 0 at every action, and a constant mean dominates
    nothing, so the front is the whole box. The run's own y frame does not exist
    yet, so the final fit's is what places that 0 in raw units.

    Raises:
        ValueError: the run recorded no fit to that many trials.
    """
    if not 0 <= trial <= len(dataset):
        raise ValueError(f'{dataset.name} records {len(dataset)} trials, not {trial}')

    objectives = dataset.get_objectives(-1 if trial == 0 else trial - 1)
    X          = plr.sample_actions(objectives.action_bounds, scan, 'sobol', seed)
    if trial == 0:
        raw_mu = objectives.to_raw(np.zeros((len(X), len(objectives.names))))

        return objectives, X, raw_mu, np.arange(len(X))

    model = dataset.get_model(trial - 1)
    if model is None:
        raise ValueError(f'{dataset.name} recorded no fit after {trial} trials')

    mu, _     = model.posterior_at(X)
    raw_mu, _ = model.posterior_at(X, raw=True)

    return model.objectives, X, raw_mu, front_order(mu, raw_mu)


def plot_action_cloud(ax, actions, colors, action_labels, bounds):
    """The action box as a cloud, for a front with no order to it. Unlike
    `plot_pareto_actions`, no path is drawn through the points."""
    ax.scatter(*actions.T, s=25, c=colors, edgecolors='black', linewidths=1,
               zorder=2, depthshade=False)
    for axis, name in zip(('x', 'y', 'z'), action_labels):
        getattr(ax, f'set_{axis}label')(name)

    for axis, b in zip(('x', 'y', 'z'), np.asarray(bounds).T):
        getattr(ax, f'set_{axis}lim')(b)

    return ax


def content_bbox(fig: plt.Figure, pad: float = 0.1):
    """The Bbox, in inches, around every axes and its labels, padded by `pad`.

    `bbox_inches='tight'` leaves 3D axis labels out, so they are added here.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes    = []
    for ax in fig.axes:
        boxes.append(ax.get_tightbbox(renderer))
        if ax.name == '3d':
            boxes += [axis.label.get_window_extent(renderer)
                      for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]

    return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)


def action_colors(objectives: plr.DecoupledObjectives, actions: np.ndarray):
    """(n, 3) RGB color of each raw action: its normalized (Hip Flex, Hip Ext,
    Delay), so each channel depends on one dimension."""
    # return 1 - objectives.xtransform(actions)
    return objectives.xtransform(actions)


def plot_color_slices(
    axes        : list[plt.Axes],
    objectives  : plr.DecoupledObjectives,
    n_pixels    : int = 128
):
    """The action colormap as (Hip Flex, Hip Ext) images at evenly spaced
    delays, highest delay in the first axes.

    Blue depends on delay alone, so each slice shows every color at its delay
    exactly, and colors between two slices are a blend of them.
    """
    (flex_lo, ext_lo, delay_lo), (flex_hi, ext_hi, delay_hi) = plr.as_bounds(objectives.action_bounds)
    flex, ext = np.meshgrid(np.linspace(flex_lo, flex_hi, n_pixels),
                            np.linspace(ext_lo, ext_hi, n_pixels))
    for ax, delay in zip(axes, np.linspace(delay_hi, delay_lo, len(axes))):
        actions = np.column_stack([flex.ravel(), ext.ravel(), np.full(flex.size, delay)])
        image   = np.clip(action_colors(objectives, actions), 0, 1).reshape(n_pixels, n_pixels, 3)
        ax.imshow(image, origin='lower', extent=(flex_lo, flex_hi, ext_lo, ext_hi), aspect='auto')
        ax.set_box_aspect(1)
        ax.set_title(f'{ACTION_LABELS[2]} = {delay * 0.005:g} sec')
        ax.set_xlabel(ACTION_LABELS[0])
        ax.set_ylabel(ACTION_LABELS[1])

    for ax in axes:
        plr.dress_axis(ax, tick_size=14, label_size=16, title_size=18, num_xticks=3, num_yticks=3)
        ax.grid(False)

    return axes


def make_figure(
    dataset   : plr.ExperimentDataset,
    path      : Path = None,
    trials    : list = [-1],
    scan      = SCAN,
    seed      = SEED,
    dpi       = DPI,
    n_slices  = SLICES
):
    if path is None:
        path = OUTPUT_DIR / (dataset.subject + f'.jpg')

    fig  = plt.figure(figsize=(5 * (len(trials) + SLICE_WIDTH), 9))
    grid = fig.add_gridspec(2, len(trials) + 1, width_ratios=[1] * len(trials) + [SLICE_WIDTH])
    obj_bounds = []
    p_axes     = []
    
    for idx, trial in enumerate(trials):
        prior      = trial == 0
        objectives, X, raw_mu, nd_idx = trial_scan(dataset, trial, scan, seed)
        names      = objectives.names
        colors     = action_colors(objectives, X)

        bounds = np.asarray([np.min(raw_mu, axis=0), np.max(raw_mu, axis=0)])
        obj_bounds.append(bounds)

        # the prior predicts the same value at every action, so the whole scan
        # lands on one point: it is drawn once, and uncolored, since no action
        # owns it
        p_ax    = fig.add_subplot(grid[0, idx])
        p_ax    = plr.plot_pareto(
            ax                      = p_ax,
            pareto                  = raw_mu[:1] if prior else raw_mu,
            nd_idx                  = np.arange(1) if prior else nd_idx,
            colors                  = '0.35' if prior else colors,
            connect                 = not prior,
            show_dominated          = not prior,
            dominated_alpha         = 0.1,
            outline_nondominated    = 1,
            nondominated_s          = 80,
            label                   = names,
            set_lims                = False
        )
        p_ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
        p_ax.set_ylabel(r'Comfort Score ($\uparrow$)')
        p_axes.append(p_ax)

        X[:, 2] *= 0.005
        bounds = objectives.action_bounds.copy()
        bounds[:, 2] *= 0.005
        a_ax    = fig.add_subplot(grid[1, idx], projection='3d')
        if prior:
            # a Sobol prefix is itself space filling, so the box is subsampled
            a_ax = plot_action_cloud(
                ax              = a_ax,
                actions         = X[:PRIOR_CLOUD],
                colors          = colors[:PRIOR_CLOUD],
                action_labels   = ACTION_LABELS,
                bounds          = bounds
            )
        else:
            a_ax = plr.plot_pareto_actions(
                ax              = a_ax,
                nd_pts          = X[nd_idx],
                colors          = colors[nd_idx],
                action_labels   = ACTION_LABELS,
                bounds          = bounds
            )
        p_ax.set_title(f'{trial} Trials')
        p_ax = plr.dress_axis(p_ax, tick_size=20, label_size=22, num_xticks=5, num_yticks=6, title_size=30)
        # p_ax.title.set_fontfamily('cmb10')   # Computer Modern bold; cmr10 has no bold weight
        a_ax = plr.dress_axis(a_ax, tick_size=20, label_size=22, num_xticks=3, num_yticks=3, num_zticks=4)
        a_ax.patch.set_visible(False)   # an opaque background covers the row above's x labels

    obj_bounds = np.asarray(obj_bounds)       # (T, 2, m): trial, [low, high], objective
    lows       = obj_bounds[:, 0].min(axis=0) # (m,) lowest value per objective over every trial
    highs      = obj_bounds[:, 1].max(axis=0) # (m,) highest
    buffer     = 0.02                          # padding, as a fraction of each objective's range
    pad        = buffer * (highs - lows)
    for ax in p_axes:
        ax.set_xlim(lows[0] - pad[0], highs[0] + pad[0])
        ax.set_ylim(lows[1] - pad[1], highs[1] + pad[1])

    # spacing between panels; the saved image is cropped to what is drawn
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.4, hspace=0.2)

    # the colormap key: n_slices delay levels stacked in the last column, spanning both rows.
    # hspace is a fraction of a slice's height, so it is solved for a gap of SLICE_GAP inches
    height = fig.get_figheight() * (fig.subplotpars.top - fig.subplotpars.bottom)
    room   = height - SLICE_GAP * (n_slices - 1)
    if room <= 0:
        raise ValueError(f'{n_slices} slices do not fit in {height:.1f} in with {SLICE_GAP} in gaps')
    key    = grid[:, -1].subgridspec(n_slices, 1, hspace=SLICE_GAP * n_slices / room)
    plot_color_slices([fig.add_subplot(key[i, 0]) for i in range(n_slices)], objectives)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=content_bbox(fig))

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'dataset', 
        type    = Path,
        nargs   = '?',
        default = DATASET,
        help    = 'the run to draw, as written by ExperimentDataset.save'
    )
    p.add_argument(
        '--output', 
        type    = Path,
        default = None,
        help    = 'the path to write the figure to'
    )
    p.add_argument(
        '--scan', 
        type    = int,
        default = SCAN,
        help    = 'number of Sobol points the front is evaluated on'
    )
    p.add_argument(
        '--seed', 
        type    = int,
        default = SEED,
        help    = 'the random seed the Sobol points are generated with'
    )
    p.add_argument(
        '--trials', 
        type    = int,
        nargs   = '+',
        default = [0, 6, 12, 18, 24],
        help    = 'how many trials each column is fit to; 0 is the prior'
    )
    p.add_argument(
        '--dpi', 
        type    = int,
        default = DPI,
        help    = 'DPI of the generated figure'
    )
    p.add_argument(
        '--slices',
        type    = int,
        default = SLICES,
        help    = 'number of delay levels the colormap key is drawn at'
    )

    return p.parse_args()


def main():
    args    = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset)
    path = make_figure(
        dataset   = dataset,
        trials    = args.trials,
        path      = args.output,
        scan      = args.scan,
        seed      = args.seed,
        dpi       = args.dpi,
        n_slices  = args.slices
    )
    print(f'Wrote to {path}')

if __name__ == '__main__':
    main()
