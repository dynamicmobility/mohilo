"""Each subject's last trial: front and set, on black, for the video.

Three columns, one per subject's most recent GP, plus the delay-color key on
the right — the same panels `human_pareto.py` draws for one subject across
trials, drawn here across subjects instead, each at its own last trial.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pypolar as plr
from scripts.icra.human_pareto import (
    ACTION_LABELS,
    SLICE_GAP,
    SLICE_WIDTH,
    SLICES,
    action_colors,
    content_bbox,
    front_order,
    plot_color_slices,
)

DATA_DIR = Path('human_data')
OUTPUT   = Path('hilo/output/subjects_pareto.jpg')

# What a subject is called in the paper, keyed by the id their run is recorded
# under — the same mapping `video/front.py`'s `SUBJECTS` uses.
SUBJECTS = {'MB02': 'Subject 1', 'MB04': 'Subject 2', 'MB05': 'Subject 3'}

SCAN = 2**14      # Sobol points the front is read off
SEED = 95
DPI  = 300

BG    = '#000000'   # figure background
INK   = '#F2F2F2'   # text, ticks, connecting lines
MUTED = '#5A6672'   # spines, grid, 3D panes


def darken(ax):
    """Recolor one already-`dress_axis`-ed axes for a black figure.

    `dress_axis` is the house style for a *light* figure, so this runs after
    it rather than replacing it: every color it set gets overridden here, and
    everything else — font, tick count, label size — is left alone. A plotted
    line defaults to black (`plot_pareto`'s connecting line, `plot_pareto_actions`'
    path), invisible on this background, so those are recolored too.
    """
    ax.set_facecolor('white')
    is_3d = ax.name == '3d'
    axes = (ax.xaxis, ax.yaxis, ax.zaxis) if is_3d else (ax.xaxis, ax.yaxis)

    for axis in axes:
        axis.label.set_color(INK)
        if is_3d:
            axis.pane.set_facecolor((0, 0, 0, 0))
            axis.pane.set_edgecolor(MUTED)
            axis.line.set_color(MUTED)
            axis._axinfo['grid']['color'] = (1, 1, 1, 0.12)
        ax.tick_params(axis=axis.axis_name, colors=MUTED, labelcolor=INK)

    if not is_3d:
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        ax.grid(True, color=INK, alpha=0.12, lw=0.8)

    ax.title.set_color(INK)
    for line in ax.get_lines():
        if line.get_color() in ('black', '#000000', (0.0, 0.0, 0.0, 1.0)):
            line.set_color(INK)

    return ax


def make_figure(
    subjects  : list[str] = tuple(SUBJECTS),
    data_dir  : Path = DATA_DIR,
    path      : Path = None,
    scan      = SCAN,
    seed      = SEED,
    dpi       = DPI,
    n_slices  = SLICES
):
    if path is None:
        path = OUTPUT

    datasets = [plr.ExperimentDataset.load(data_dir / s / f'{s}.json') for s in subjects]

    fig  = plt.figure(figsize=(5 * (len(subjects) + SLICE_WIDTH), 9), facecolor=BG)
    grid = fig.add_gridspec(2, len(subjects) + 1, width_ratios=[1] * len(subjects) + [SLICE_WIDTH])

    for idx, dataset in enumerate(datasets):
        # Each subject's own last trial, mirroring `human_pareto.py`'s single
        # `trials=[-1]` panel — here one such panel per subject instead of
        # one subject across several trials.
        model = dataset.get_model(-1)
        names = model.objectives.names
        X     = plr.sample_actions(model.action_bounds, scan, 'sobol', seed)

        mu, std         = model.posterior_at(X)
        raw_mu, raw_std = model.posterior_at(X, raw=True)
        nd_idx          = front_order(mu, raw_mu)
        colors          = action_colors(model.objectives, X)

        p_ax = fig.add_subplot(grid[0, idx])
        p_ax = plr.plot_pareto(
            ax                    = p_ax,
            pareto                = raw_mu,
            nd_idx                = nd_idx,
            colors                = colors,
            connect               = True,
            show_dominated        = True,
            dominated_alpha       = 0.1,
            outline_nondominated  = 1,
            nondominated_s        = 80,
            label                 = names,
            # Each subject explored its own range of the two objectives, unlike
            # `human_pareto.py`'s own trials of one run, so a panel is zoomed to
            # its own front rather than sharing one box across subjects.
            set_lims              = False
        )
        p_ax.set_xlabel(r'Metabolic Cost (W/kg, $\downarrow$)')
        p_ax.set_ylabel(r'Comfort Score ($\uparrow$)')
        p_ax.set_title(SUBJECTS[dataset.subject])

        a_ax = fig.add_subplot(grid[1, idx], projection='3d')
        a_ax = plr.plot_pareto_actions(
            ax              = a_ax,
            nd_pts          = X[nd_idx],
            colors          = colors[nd_idx],
            action_labels   = ACTION_LABELS,
            bounds          = model.objectives.action_bounds,
        )
        p_ax = plr.dress_axis(p_ax, tick_size=20, label_size=22, num_xticks=5,
                              num_yticks=6, title_size=30)
        a_ax = plr.dress_axis(a_ax, tick_size=20, label_size=22, num_xticks=4, num_yticks=4,
                              num_zticks=4)
        darken(p_ax)
        darken(a_ax)

    # spacing between panels; the saved image is cropped to what is drawn
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.4, hspace=0.2)

    # the colormap key: n_slices delay levels stacked in the last column, spanning both rows.
    # hspace is a fraction of a slice's height, so it is solved for a gap of SLICE_GAP inches
    height = fig.get_figheight() * (fig.subplotpars.top - fig.subplotpars.bottom)
    room   = height - SLICE_GAP * (n_slices - 1)
    if room <= 0:
        raise ValueError(f'{n_slices} slices do not fit in {height:.1f} in with {SLICE_GAP} in gaps')
    key         = grid[:, -1].subgridspec(n_slices, 1, hspace=SLICE_GAP * n_slices / room)
    slice_axes  = [fig.add_subplot(key[i, 0]) for i in range(n_slices)]
    plot_color_slices(slice_axes, model.objectives)
    for ax in slice_axes:
        darken(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=content_bbox(fig), facecolor=BG)

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        'subjects',
        nargs   = '*',
        default = list(SUBJECTS),
        help    = 'subject names, each read from <data-dir>/<subject>/<subject>.json'
    )
    p.add_argument(
        '--data-dir',
        type    = Path,
        default = DATA_DIR,
        help    = 'the folder holding one folder per subject'
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
    args = parse_args()
    path = make_figure(
        subjects  = args.subjects,
        data_dir  = args.data_dir,
        path      = args.output,
        scan      = args.scan,
        seed      = args.seed,
        dpi       = args.dpi,
        n_slices  = args.slices
    )
    print(f'Wrote to {path}')


if __name__ == '__main__':
    main()
