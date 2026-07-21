import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import pypolar as plr


def load_run(run_dir):
    """Load a single run folder produced by experiment.py."""
    run_dir = Path(run_dir)
    with np.load(run_dir / 'data.npz') as npz:
        data = {k: npz[k] for k in npz.files}
    meta_path = run_dir / 'run.json'
    data['meta'] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return data


def make_gif(run_dir, savepath=None, fps=3, tol=0.0):
    """Animate a run to a gif as a 2x2 grid.

    Top row: the two GPs (one per objective) with mean +/- std, the groundtruth
    objective curve, and every feedback point collected through iteration i.
    Bottom row: the groundtruth hypervolume and pareto overlay over iterations,
    computed from each iteration's GP mean, tracing out up to iteration i.
    """
    run_dir = Path(run_dir)
    data = load_run(run_dir)

    mu = data['mu']                    # (n_iters, num_objs, N)
    std = data['std']                  # (n_iters, num_objs, N)
    feedback = data['feedback']        # (n_iters, num_objs + 1): [idx, *vals]
    action_space = data['action_space']  # (N, d)
    true_objs = data['true_objs']      # (N, num_objs)

    if action_space.shape[1] != 1:
        raise ValueError(
            f'plot_run only supports 1D action spaces; got shape {action_space.shape}. '
        )

    n_iters, num_objs, _ = mu.shape
    if num_objs != 2:
        raise ValueError(
            f'The 2x2 layout expects 2 objectives; got num_objs={num_objs}.'
        )

    x = action_space[:, 0]
    order = np.argsort(x)
    xs = x[order]

    fb_idx = feedback[:, 0].astype(int)   # action index of each feedback point
    fb_x = x[fb_idx]                      # action value of each feedback point
    fb_vals = feedback[:, 1:]            # (n_iters, num_objs) observed values

    # Per-iteration performance, computed from each iteration's GP mean
    hvs = np.array([plr.groundtruth_hypervolume(mu[i].T, true_objs, tol=tol) for i in range(n_iters)])
    overlays = np.array([plr.pareto_overlay(mu[i].T, true_objs, tol=tol) for i in range(n_iters)])
    iters = np.arange(1, n_iters + 1)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
    gp_axes = axs[0]                     # top row: the two GPs
    hv_ax, ov_ax = axs[1]                # bottom row: hv and overlay

    colors = plt.cm.tab10(np.arange(num_objs))

    # Stable y-limits across the whole animation, per objective
    ylims = []
    for j in range(num_objs):
        lo = min((mu[:, j] - std[:, j]).min(), true_objs[:, j].min(), fb_vals[:, j].min())
        hi = max((mu[:, j] + std[:, j]).max(), true_objs[:, j].max(), fb_vals[:, j].max())
        pad = 0.05 * (hi - lo + 1e-9)
        ylims.append((lo - pad, hi + pad))

    w = data['meta'].get('config', {}).get('objective', {}).get('w')

    def draw_metric(ax, series, title, ylabel, color, i):
        ax.clear()
        # Full curve faint, traced portion solid, current point highlighted
        ax.plot(iters, series, color=color, alpha=0.2, lw=1.5)
        ax.plot(iters[:i + 1], series[:i + 1], color=color, lw=2.5)
        ax.scatter(iters[i], series[i], color='red', edgecolor='k', zorder=5, s=60)
        ax.set_xlim(iters.min(), iters.max())
        lo, hi = series.min(), series.max()
        pad = 0.05 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def draw(i):
        for j, ax in enumerate(gp_axes):
            ax.clear()
            m = mu[i, j][order]
            s = std[i, j][order]

            # Groundtruth objective curve
            ax.plot(xs, true_objs[:, j][order], color='k', ls='--',
                    lw=1.5, label='groundtruth')

            # GP mean +/- std
            ax.fill_between(xs, m - s, m + s, color=colors[j], alpha=0.25)
            ax.plot(xs, m, color=colors[j], lw=2.5, label='GP mean')

            # Feedback collected through iteration i
            ax.scatter(fb_x[:i + 1], fb_vals[:i + 1, j], color=colors[j],
                       edgecolor='k', zorder=5, s=40, label='feedback')
            # Highlight the newest feedback point
            ax.scatter(fb_x[i], fb_vals[i, j], color='red', edgecolor='k',
                       zorder=6, s=70, label='newest')

            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(*ylims[j])
            ax.set_xlabel('action')
            ax.set_ylabel(f'objective {j}')
            ax.set_title(f'Objective {j}')
            if j == 0:
                ax.legend(loc='best', fontsize=8)

        draw_metric(hv_ax, hvs, 'Groundtruth hypervolume', 'hypervolume', 'tab:blue', i)
        draw_metric(ov_ax, overlays, 'Pareto overlay', 'overlay', 'tab:green', i)

        title = f'Iteration {i + 1}/{n_iters}'
        if w is not None:
            title += f'   w={np.asarray(w).ravel().tolist()}'
        fig.suptitle(title)

    anim = FuncAnimation(fig, draw, frames=n_iters, interval=1000 / fps)

    if savepath is None:
        savepath = run_dir / 'gp_learning.gif'
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    anim.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'Saved gif to {savepath.resolve()}')
    return savepath


def parse_args():
    parser = argparse.ArgumentParser(
        description='Animate the GPs learned over iterations for a single run.'
    )
    parser.add_argument('run_dir', type=Path,
                        help='Path to a run_XXX folder (containing data.npz).')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output gif path (default: <run_dir>/gp_learning.gif).')
    parser.add_argument('--fps', type=float, default=3, help='Frames per second.')
    parser.add_argument('--tol', type=float, default=0.0,
                        help='Non-domination tolerance for the hv/overlay curves, as a '
                             'fraction of each objective range (0 = strict).')
    return parser.parse_args()


def main():
    args = parse_args()
    make_gif(args.run_dir, savepath=args.out, fps=args.fps, tol=args.tol)


if __name__ == '__main__':
    main()
