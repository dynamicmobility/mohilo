"""One run, animated: every trial's fit and the acquisition that chose from it.

A dataset stores the fits themselves rather than only the measurements, so each
frame is the posterior the run actually queried, reloaded rather than refit.
"""

# TODO: clean up this file (and maybe move functionality to package?)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

import pypolar as plr
from pypolar.optimization.gp import DTYPE

GRID_POINTS = 1024        # points both panels are drawn over
NUM_PATHS   = 16          # posterior samples per frame
FPS         = 1.5
DPI         = 120
ACQ_COLOR   = '#0072B2'   # Okabe-Ito blue, the one plot_fit_1d leaves free
DATASET     = Path('hilo/output/experiments/20260831_061404/test.json')


def action_grid(box: np.ndarray, n: int = GRID_POINTS): # TODO: can this be replaced with the objective bounds?
    return np.linspace(box[0, 0], box[1, 0], n)[:, None]


def plot_fit(
    ax         : plt.Axes,
    truth      : plr.SyntheticOracle,
    objective  : plr.Objective,
    gp         : plr.BoTorchGP,
    recommended: np.ndarray,
    box        : np.ndarray,
):
    """The 1D fit: the GP and the truth evaluated on a grid, drawn by
    `plr.plot_fit_1d`.
    """
    grid    = action_grid(box)
    mu, std = gp.posterior_at(grid, raw=True)
    vlines  = {'true optimizer': truth.truth.optimizers[0, 0].item()}
    if recommended is not None:
        vlines = {'recommended action': recommended[0], **vlines}

    plr.plot_fit_1d(
        ax       = ax,
        x        = grid[:, 0],
        mu       = mu[:, 0],
        std      = std[:, 0],
        xdata    = objective.xdata[:, 0],
        ydata    = objective.ydata,
        # a SyntheticFunction, so `noise=False` is the noiseless truth
        truth    = truth(grid, noise=False),
        # one objective, so the single column is the path itself
        paths    = gp.sample_paths(grid, NUM_PATHS, raw=True)[:, :, 0],
        vlines   = vlines,
        band_std = 1.0,
        title    = f'{type(truth.truth).__name__}, {truth.truth.dim}D, '
                   f'after {len(objective.ydata)} measurements'
    )


def acquisition_at(acqf, gp: plr.BoTorchGP, X: np.ndarray):
    """The acquisition `acqf` at the (n, d) raw actions X, returned (n,)."""
    # the search runs in the GP's frame, and botorch reads (batch, q, d), so
    # every action goes in normalized and as its own q = 1 candidate
    with torch.no_grad():
        return acqf(torch.as_tensor(gp.frame(X), dtype=DTYPE).unsqueeze(1)).numpy()


def plot_acquisition(
    ax     : plt.Axes,
    acqf   : plr.AcquisitionFunction,
    gp     : plr.BoTorchGP,
    action : np.ndarray,
    box    : np.ndarray,
    label  : str = None,
):
    """The acquisition surface this trial maximized, and the action it returned.

    An acquisition is a function of the GP, so it is rebuilt against this
    trial's fit; `_incumbent` supplies the measurement-dependent arguments its
    class takes -- an incumbent value, or the baseline points.
    """
    grid   = action_grid(box)
    fn     = acqf.acqf(gp.model, **acqf._incumbent(gp))
    values = acquisition_at(fn, gp, grid)

    ax.plot(grid[:, 0], values, color=ACQ_COLOR, lw=1.5, zorder=2)
    ax.fill_between(grid[:, 0], values.min(), values, color=ACQ_COLOR,
                    alpha=0.15, zorder=1)
    if action is not None:
        ax.axvline(action[0], color=ACQ_COLOR, ls='-.', lw=1.2, zorder=3,
                   label='next action')
        ax.scatter(action[0], acquisition_at(fn, gp, np.atleast_2d(action)),
                   s=28, color=ACQ_COLOR, zorder=4)
        ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax.set_xlabel('action')
    ax.set_ylabel(label or 'acquisition')
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)

    return ax


def make_gif(dataset: plr.ExperimentDataset, path: Path = None, fps: float = FPS):
    """Every fitted trial of a run as one frame, written to a gif.

    A frame needs a fit, so the opening trial -- whose action was drawn at
    random, before any measurement -- has none and is skipped.

    Args:
        dataset: the run to draw.
        path: the gif to write. Defaults to the dataset's own path, `.gif`.
        fps: frames per second.

    Returns:
        Where the gif went.
    """
    frames = [record for record in dataset if record.state_dict is not None]
    if not frames:
        raise ValueError(f'{dataset.name} has no fitted trial to draw')

    path = Path(path or dataset.path.with_suffix('.gif'))
    path.parent.mkdir(parents=True, exist_ok=True)

    last  = dataset.get_objective(frames[-1].trial)
    truth = dataset.get_groundtruth()
    if last.action_dim != 1:
        # botorch would raise on the grid instead, several frames in
        raise ValueError(f'both panels draw one action, got {last.action_dim}D')

    box = np.broadcast_to(
        np.asarray(last.action_box(), dtype=float).reshape(2, -1),
        (2, last.action_dim)
    )
    # one y range for every frame, so the truth does not jump as the gif runs.
    # An early band is far wider than this and clips, which is the point: the
    # frames are only comparable on a fixed axis.
    span = np.concatenate([truth(action_grid(box), noise=False), last.ydata])
    pad  = 0.25 * np.ptp(span)

    fig, (ax_fit, ax_acq) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True, height_ratios=[3, 2],
        constrained_layout=True
    )
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=DPI):
        for record in frames:
            ax_fit.clear()
            ax_acq.clear()
            gp = dataset.get_model(record.trial)
            plot_fit(
                ax          = ax_fit,
                truth       = truth,
                objective   = gp.objective,
                gp          = gp,
                recommended = record.aux.get('recommended'),
                box         = box,
            )
            plot_acquisition(
                ax     = ax_acq,
                acqf   = dataset.get_acquisition(),
                gp     = gp,
                action = record.action,
                box    = box,
                label  = f'{dataset.acquisition.strategy} acquisition',
            )
            ax_fit.set_xlabel('')
            ax_fit.set_ylim(span.min() - pad, span.max() + pad)
            ax_fit.set_xlim(box[0, 0], box[1, 0])
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
