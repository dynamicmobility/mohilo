"""One run's inferred Pareto front, animated: every fitted trial as a frame, in
objective space beside the actions that produced it.

A dataset stores the fits themselves, so each frame is the posterior that trial
actually held, not a refit. Every frame is drawn in one axis box, shared across
the gif, so the front's movement between trials is the only thing that moves.
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


def front_order(mu, raw_mu):
    """Sorted nondominated indexes of the Pareto front."""
    nd_idx = plr.get_nondominated_tol(mu)

    return nd_idx[np.argsort(raw_mu[nd_idx, 0])]


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


def make_figure(
    dataset   : plr.ExperimentDataset,
    v_dataset : plr.ExperimentDataset,
    path      : Path = None,
    trial     : int = [-1],
    scan      = SCAN,
    seed      = SEED,
    dpi       = DPI
):
    if path is None:
        path = OUTPUT_DIR / (dataset.subject + '_validation' + f'.jpg')

    fig  = plt.figure(figsize=(5, 5))
    obj_bounds = []
    p_axes     = []

    model   = dataset.get_model(trial)
    names   = model.objectives.names
    X       = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)

    mu, std           = model.posterior_at(X)
    raw_mu, raw_std   = model.posterior_at(X, raw=True)
    nd_idx            = front_order(mu, raw_mu)
    colors            = model.objectives.xtransform(X) # normalization space

    bounds = np.asarray([np.min(raw_mu, axis=0), np.max(raw_mu, axis=0)])
    obj_bounds.append(bounds)

    p_ax    = fig.add_subplot(1, 1, 1)
    p_ax    = plr.plot_pareto(
        ax                      = p_ax,
        pareto                  = raw_mu,
        nd_idx                  = nd_idx,
        colors                  = colors,
        connect                 = True,
        show_dominated          = True,
        dominated_alpha         = 0.4,
        outline_nondominated    = 2,
        nondominated_s          = 30,
        label                   = names,
        set_lims                = False
    )
    X = v_dataset.get_actions()
    valid_mu, valid_std   = model.posterior_at(X, raw=True)
    p_ax.scatter(valid_mu[:, 0], valid_mu[:, 1], marker='s', s=100, color='royalblue', edgecolor='black')
    v_dataset.get_objectives(-1)

    p_ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    p_ax.set_ylabel(r'Comfort Score ($\uparrow$)')
    p_axes.append(p_ax)
    p_ax.set_title(f'Trial {trial}')
    p_ax = plr.dress_axis(p_ax, label_size=16, num_xticks=5, num_yticks=6, title_size=18)

    # a_ax    = fig.add_subplot(2, len(trials), (idx + 1) + len(trials), projection='3d')
    # a_ax = plr.plot_pareto_actions(
    #     ax              = a_ax,
    #     nd_pts          = X[nd_idx],
    #     colors          = colors[nd_idx],
    #     action_labels   = ['Hip Flexion Scale', 'Hip Extension Scale', 'Delay'],
    #     bounds          = model.objectives.action_bounds
    # )
    # a_ax = plr.dress_axis(a_ax, label_size=16, num_xticks=4, num_yticks=4, num_zticks=4)
    # a_ax.patch.set_visible(False)   # an opaque background covers the row above's x labels

    # obj_bounds = np.asarray(obj_bounds)       # (T, 2, m): trial, [low, high], objective
    # lows       = obj_bounds[:, 0].min(axis=0) # (m,) lowest value per objective over every trial
    # highs      = obj_bounds[:, 1].max(axis=0) # (m,) highest
    # buffer     = 0.02                          # padding, as a fraction of each objective's range
    # pad        = buffer * (highs - lows)
    # for ax in p_axes:
    #     ax.set_xlim(lows[0] - pad[0], highs[0] + pad[0])
    #     ax.set_ylim(lows[1] - pad[1], highs[1] + pad[1])

    # spacing between panels; the saved image is cropped to what is drawn
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.4, hspace=0.2)
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
        '--trial', 
        type    = int,
        default = -1,
        help    = 'the trial to plot'
    )
    p.add_argument(
        '--dpi', 
        type    = int,
        default = DPI,
        help    = 'DPI of the generated figure'
    )

    return p.parse_args()


def main():
    args    = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset / (str(args.dataset.name) + '.json'))
    v_dataset = plr.ExperimentDataset.load(args.dataset / 'evaluation.json')
    path = make_figure(
        dataset   = dataset,
        v_dataset = v_dataset,
        trial     = args.trial,
        path      = args.output,
        scan      = args.scan,
        seed      = args.seed,
        dpi       = args.dpi
    )
    print(f'Wrote to {path}')

if __name__ == '__main__':
    main()
