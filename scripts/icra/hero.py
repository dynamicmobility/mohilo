"""One subject's final inferred Pareto front, drawn in objective space and in
action space as two separate figures, plus the action colormap key as a third.

Each subject's run is `<data-dir>/<subject>/<subject>.json`. The front is read
off a Sobol scan of the last trial's GP, colored by action as in human_pareto.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import pypolar as plr
from scripts.icra.human_pareto import (ACTION_LABELS, SLICE_GAP, action_colors, content_bbox,
                                       front_order, plot_color_slices)

SUBJECT     = 'MB04'
DATA_DIR    = Path('human_data')
OUTPUT_DIR  = Path('hilo/output/')

TRIAL       = -1        # the optimization trial whose GP is drawn
SCAN        = 2**14     # Sobol points the front is read off
SEED        = 95
DPI         = 300
SLICES      = 3         # delay levels the colormap key is drawn at


def save(fig: plt.Figure, path: Path, dpi: int = DPI):
    """Write `fig` to `path`, cropped to its content, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=content_bbox(fig))
    plt.close(fig)

    return path


def make_figures(
    dataset     : plr.ExperimentDataset,
    output_dir  : Path = OUTPUT_DIR,
    trial       : int  = TRIAL,
    scan        : int  = SCAN,
    seed        : int  = SEED,
    dpi         : int  = DPI,
    n_slices    : int  = SLICES
):
    """The objective-space front, its actions and the colormap key of one
    trial, written to `output_dir`. Returns the three paths written."""
    model   = dataset.get_model(trial)
    X       = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)

    mu, _       = model.posterior_at(X)
    raw_mu, _   = model.posterior_at(X, raw=True)
    nd_idx      = front_order(mu, raw_mu)
    colors      = action_colors(model.objectives, X)

    p_fig = plt.figure(figsize=(5, 4.5))
    p_ax  = plr.plot_pareto(
        ax                      = p_fig.add_subplot(),
        pareto                  = raw_mu,
        nd_idx                  = nd_idx,
        colors                  = colors,
        connect                 = True,
        show_dominated          = True,
        dominated_alpha         = 0.1,
        outline_nondominated    = 1,
        nondominated_s          = 80,
        label                   = model.objectives.names,
        set_lims                = False
    )
    p_ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
    p_ax.set_ylabel(r'Comfort Score ($\uparrow$)')
    plr.dress_axis(p_ax, tick_size=20, label_size=22, num_xticks=5, num_yticks=6, title_size=30)

    X[:, 2] *= 0.005
    bounds = model.objectives.action_bounds.copy()
    bounds[:, 2] *= 0.005
    a_fig = plt.figure(figsize=(5, 4.5))
    a_ax  = plr.plot_pareto_actions(
        ax              = a_fig.add_subplot(projection='3d'),
        nd_pts          = X[nd_idx],
        colors          = colors[nd_idx],
        action_labels   = ACTION_LABELS,
        bounds          = bounds
    )
    plr.dress_axis(a_ax, tick_size=20, label_size=22, num_xticks=3, num_yticks=3, num_zticks=4)

    # hspace is a fraction of a slice's height, so it is solved for a gap of SLICE_GAP inches
    k_fig  = plt.figure(figsize=(3, 3 * n_slices + SLICE_GAP * (n_slices - 1)))
    height = k_fig.get_figheight() * (k_fig.subplotpars.top - k_fig.subplotpars.bottom)
    room   = height - SLICE_GAP * (n_slices - 1)
    k_fig.subplots_adjust(hspace=SLICE_GAP * n_slices / room)
    plot_color_slices(list(k_fig.subplots(n_slices, 1)), model.objectives)

    name = dataset.subject
    return (save(p_fig, output_dir / f'{name}_objectives.jpg', dpi),
            save(a_fig, output_dir / f'{name}_actions.jpg', dpi),
            save(k_fig, output_dir / f'{name}_key.jpg', dpi))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'subject',
        nargs   = '?',
        default = SUBJECT,
        help    = 'subject name, read from <data-dir>/<subject>/<subject>.json'
    )
    p.add_argument(
        '--data-dir',
        type    = Path,
        default = DATA_DIR,
        help    = 'the folder holding one folder per subject'
    )
    p.add_argument(
        '--output-dir',
        type    = Path,
        default = OUTPUT_DIR,
        help    = 'the folder the three figures are written to'
    )
    p.add_argument(
        '--trial',
        type    = int,
        default = TRIAL,
        help    = 'the optimization trial whose GP is drawn'
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
        '--dpi',
        type    = int,
        default = DPI,
        help    = 'DPI of the generated figures'
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
    dataset = plr.ExperimentDataset.load(args.data_dir / args.subject / f'{args.subject}.json')
    paths   = make_figures(
        dataset     = dataset,
        output_dir  = args.output_dir,
        trial       = args.trial,
        scan        = args.scan,
        seed        = args.seed,
        dpi         = args.dpi,
        n_slices    = args.slices
    )
    for path in paths:
        print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
