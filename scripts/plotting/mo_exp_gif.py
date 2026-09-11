"""One multi-objective run, animated: the true Pareto front and the front the
model infers, trial by trial, in the truth's own objective space.

The actions are 2D, so no fit or acquisition surface is drawn -- only where the
inferred front lands relative to the real one. A dataset stores the fits
themselves, so each frame is the posterior the run actually queried.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

import pypolar as plr
from pypolar.utils.pareto import get_nondominated
from pypolar.utils.plotting import plot_mo_space

FPS      = 1.5
DPI      = 120
DATASET  = Path('hilo/output/experiments/20260831_061404/test.json')


def measured_at(gp: plr.DecoupledMOGP, truth: plr.MOSyntheticOracle):
    """(N, m) true values at every action the run has measured. Decoupled, so
    each objective carries its own actions and the run has visited their union.
    """
    actions = np.vstack([gp.objectives[i].xdata for i in range(len(gp.objectives))])

    return truth(actions, noise=False)


def inferred_front(gp: plr.DecoupledMOGP, truth: plr.MOSyntheticOracle):
    """(k, m) true values at the model's inferred Pareto set.

    `posterior_at` returns maximization space, which is `get_nondominated`'s own
    larger-is-better convention, so the set is read off it with no sign
    handling; only the action indices cross over to the truth.
    """
    mu, _ = gp.posterior_at(truth.scan_actions)

    return truth.scan_values[get_nondominated(mu)]


def true_front(truth: plr.MOSyntheticOracle, objective: plr.DecoupledObjectives):
    """(k, m) the truth's own front over its scan, sorted by the first
    objective so it draws as a line. BoTorch states a truth in the minimizing
    sense, so the values go through the objectives' own transforms first.
    """
    # each column through its own objective's transform, so the front is read
    # in the same maximization space the model's is
    front = truth.scan_values[get_nondominated(objective.ytransform(truth.scan_values))]

    return front[np.argsort(front[:, 0])]


def make_gif(dataset: plr.ExperimentDataset, path: Path = None, fps: float = FPS):
    """Every fitted trial of a run as one frame, written to a gif.

    The opening trials are chosen at random, before any fit, so they have no
    inferred front and are skipped.

    Args:
        dataset: the run to draw.
        path: the gif to write. Defaults to the dataset's own path, `.gif`.
        fps: frames per second.

    Returns:
        Where the gif went.
    """
    frames = [record for record in dataset.trials if record.state_dict is not None]
    if not frames:
        raise ValueError(f'{dataset.name} has no fitted trial to draw')

    path = Path(path or dataset.path.with_suffix('.gif'))
    path.parent.mkdir(parents=True, exist_ok=True)

    truth = dataset.get_groundtruth()
    front = true_front(truth, dataset.get_objectives())
    names = dataset.groundtruth.objectives

    # one axis box for every frame, so the true front does not move as the gif
    # runs. It spans the front and the reference point, which is the region
    # hypervolume is measured over: a measurement beyond the reference is worse
    # than the worst value that counts, and falls outside the view
    # span = np.vstack([front, truth.ref_point])
    # low  = span.min(axis=0) - 0.05 * np.ptp(span, axis=0)
    # high = span.max(axis=0) + 0.05 * np.ptp(span, axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    writer  = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=DPI):
        for record in frames:
            gp        = dataset.get_model(record.trial)
            measured  = measured_at(gp, truth)
            predicted = inferred_front(gp, truth)

            ax.clear()
            plot_mo_space(
                ax,
                true_front = front,
                measured   = measured,
                predicted  = predicted,
                names      = [f'{name} (lower is better)' for name in names],
                title      = f'{dataset.groundtruth.func}, {dataset.name}, '
                             f'trial {record.trial}'
            )
            # ax.set_xlim(low[0], high[0])
            # ax.set_ylim(low[1], high[1])
            writer.grab_frame()

    plt.close(fig)

    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset', type=Path, nargs='?', default=DATASET,
                   help='the run to draw, as written by ExperimentDataset.save')
    p.add_argument('--output', type=Path, default=None,
                   help="the gif to write; defaults to the dataset's own path, .gif")
    p.add_argument('--fps',    type=float, default=FPS, help='frames per second')

    return p.parse_args()


def main():
    args = parse_args()
    dataset = plr.ExperimentDataset.load(args.dataset)
    print(f'wrote {make_gif(dataset, path=args.output, fps=args.fps)}')


if __name__ == '__main__':
    main()
