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

import pypolar as plr

DATASET     = Path('hilo/output/experiments/Aug28_Neil/MT01.json')
OUTPUT_DIR  = Path('hilo/output/')

SCAN       = 2**14      # Sobol points the front is read off
SEED       = 95
DPI        = 120


def front_order(mu, raw_mu):
    """Sorted nondominated indexes of the Pareto front."""
    nd_idx = plr.get_nondominated_tol(mu)

    return nd_idx[np.argsort(raw_mu[nd_idx, 0])]


def make_figure(
    dataset   : plr.ExperimentDataset,
    path      : Path = None,
    trial     : int = -1,
    scan      = SCAN,
    seed      = SEED,
    dpi       = DPI
):
    if path is None:
        path = OUTPUT_DIR / (dataset.subject + f'trial{trial}.jpg')
    model   = dataset.get_model(trial)
    names   = model.objectives.names
    X       = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)

    mu, std           = model.posterior_at(X)
    raw_mu, raw_std   = model.posterior_at(X, raw=True)
    nd_idx            = front_order(mu, raw_mu)
    colors            = model.objectives.xtransform(X) # normalization space

    fig  = plt.figure(figsize=(12, 5), constrained_layout=True)
    p_ax = fig.add_subplot(1, 2, 1)
    a_ax = fig.add_subplot(1, 2, 2, projection='3d')
    path.parent.mkdir(parents=True, exist_ok=True)
    
    p_ax = plr.plot_pareto(
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
    p_ax.set_xlabel('Metabolic Cost (W/kg, ↓)')
    p_ax.set_ylabel('Comfort Score (↑)')

    a_ax = plr.plot_pareto_actions(
        ax              = a_ax,
        nd_pts          = X[nd_idx],
        colors          = colors[nd_idx],
        action_labels   = ['Hip Flexion Scale', 'Hip Extension Scale', 'Delay'],
        bounds          = model.objectives.action_bounds
    )
    p_ax = plr.dress_axis(p_ax)
    a_ax = plr.dress_axis(a_ax)
    fig.savefig(path, dpi=dpi)

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
    dataset = plr.ExperimentDataset.load(args.dataset)
    path = make_figure(
        dataset   = dataset,
        trial     = args.trial,
        path      = args.output,
        scan      = args.scan,
        seed      = args.seed,
        dpi       = args.dpi
    )
    print(f'Wrote to {path}')

if __name__ == '__main__':
    main()
