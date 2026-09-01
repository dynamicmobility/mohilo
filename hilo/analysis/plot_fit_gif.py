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

DATASET = Path('hilo/output/experiments/Aug28_Neil/MT01.json')
OUTPUT  = Path('hilo/output/front_samples.gif')

SCAN       = 2**14      # Sobol points the front is read off
SEED       = 95
FPS        = 1.5
DPI        = 120
PAD        = 0.05       # margin on the objective axes, as a fraction of the span
FRONT_CMAP = 'RdYlGn'   # front points colored by their position along it
ACTION_NAMES = ('h_flex_torque_scale', 'h_ext_torque_scale', 'hip_delay_idx')


def front_order(mu, raw_mu):
    """(k,) indices of the non-dominated set, sorted by the first objective so
    the front draws as a line.

    Args:
        mu: (n, m) posterior means in maximization space, which is what
            dominance is read in.
        raw_mu: (n, m) the same means in the objectives' own units, which is
            what is plotted and therefore what the ordering must follow.
    """
    nd_idx = plr.get_nondominated_tol(mu)

    return nd_idx[np.argsort(raw_mu[nd_idx, 0])]


def measured_pairs(mogp):
    """(N, 2) the two objectives' measured values, paired by action.

    The objectives are decoupled, so each carries its own `xdata`; a pair is
    formed by matching every action of the second objective to the nearest
    action of the first.
    """
    actions0 = mogp.objectives[0].xdata
    actions1 = mogp.objectives[1].xdata
    idxs = np.array([np.argmin(np.linalg.norm(actions0 - a, axis=1)) for a in actions1])

    return np.column_stack([mogp.objectives[0].ydata[idxs], mogp.objectives[1].ydata])


def padded_limits(points, pad=PAD):
    """(low, high) per column, widened by `pad` of each column's span."""
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = pad * np.ptp(points, axis=0)

    return low - margin, high + margin

def draw_frame(p_ax, a_ax, X, mu, raw_mu, measured, names, mogp: plr.DecoupledMOGP):
    """One trial's objective space and action space, on axes already cleared.

    Args:
        X: (n, d) the scanned actions.
        mu: (n, m) posterior means in maximization space.
        raw_mu: (n, m) the same means in the objectives' own units.
        measured: (N, 2) paired measurements, in raw units.
        names: the two objectives' names.
    """
    nd_idx = front_order(mu, raw_mu)
    colors = mogp.objectives.xtransform(X)
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
        label                   = names
    )
    a_ax = plr.plot_pareto_actions(
        ax              = a_ax,
        nd_pts          = X[nd_idx],
        colors          = colors[nd_idx],
        action_labels   = ['Hip Flexion Scale', 'Hip Extension Scale', 'Delay']
    )

    plr.dress_axis(p_ax)
    plr.dress_axis(a_ax)



def make_gif(dataset: plr.ExperimentDataset, path=None, fps=FPS, scan=SCAN, seed=SEED):
    """Every fitted trial of a run as one frame, written to a gif.

    The opening trials are chosen before any fit, so they carry no posterior
    and are skipped. The posteriors are read first and drawn second: the axis
    limits must hold every frame's points, so they cannot be known until all of
    them are computed.

    Args:
        dataset: the `ExperimentDataset` to draw.
        path: the gif to write. Defaults to `OUTPUT`.
        fps: frames per second.
        scan: Sobol points the front is read off.
        seed: the scan's seed.

    Returns:
        Where the gif went.
    """
    frames = [record.trial for record in dataset.trials
              if record.state_dict is not None]
    if not frames:
        raise ValueError(f'{dataset.name} has no fitted trial to draw')

    models = [dataset.get_model(trial) for trial in frames]
    names  = models[-1].objectives.names
    X = plr.sample_actions(models[-1].action_bounds, scan, 'sobol', seed)

    posteriors = [(mogp.posterior_at(X)[0],            # maximization space
                   mogp.posterior_at(X, raw=True)[0],  # objectives' own units
                   measured_pairs(mogp))
                  for mogp in models]

    # one axis box for the whole gif, spanning every point any frame draws
    low, high = padded_limits(np.vstack(
        [raw_mu for _, raw_mu, _ in posteriors]
        + [measured for _, _, measured in posteriors]
    ))
    action_low, action_high = X.min(axis=0), X.max(axis=0)

    fig  = plt.figure(figsize=(12, 5), constrained_layout=True)
    p_ax = fig.add_subplot(1, 2, 1)
    a_ax = fig.add_subplot(1, 2, 2, projection='3d')

    path = Path(path or OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=DPI):
        for trial, (mu, raw_mu, measured) in zip(frames, posteriors):
            p_ax.clear()
            a_ax.clear()
            try:
                draw_frame(p_ax, a_ax, X, mu, raw_mu, measured, names, dataset.get_model(trial=trial))
            except Exception as e:
                print(e)
                quit()

            p_ax.set_xlim(low[0], high[0])
            p_ax.set_ylim(low[1], high[1])
            a_ax.set_xlim(action_low[0], action_high[0])
            a_ax.set_ylim(action_low[1], action_high[1])
            a_ax.set_zlim(action_low[2], action_high[2])
            p_ax.set_title(f'{dataset.name}, trial {trial}')
            writer.grab_frame()

    plt.close(fig)

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path, nargs='?', default=DATASET,
                   help='the run to draw, as written by ExperimentDataset.save')
    p.add_argument('--output', type=Path, default=OUTPUT, help='the gif to write')
    p.add_argument('--fps',    type=float, default=FPS, help='frames per second')
    p.add_argument('--scan',   type=int, default=SCAN,
                   help='Sobol points the front is read off')

    return p.parse_args()


def main():
    args    = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset)
    print(f'wrote {make_gif(dataset, path=args.output, fps=args.fps, scan=args.scan)}')


if __name__ == '__main__':
    main()
