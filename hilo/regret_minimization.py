"""Sequential Bayesian optimization on a synthetic objective, scored two ways
against the strategy that chose the points: regret in objective value, and
distance in action space to the true optimizer.
"""

import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler

import pypolar as plr

DIM          = 3        # action dimension
NUM_STEPS    = 15       # acquisition steps taken
NOISE        = 0.5      # noise added, as a fraction of the truth's spread
GP_NOISE     = plr.NoiseModel.pinned(0.5)  # a float pins, None fits, prior() regularizes
MIN_LENGTHSCALE = None   # lengthscale floor, at the design spacing
STRATEGY     = 'lognei'  # one of STRATEGIES; 'random' is the non-adaptive baseline
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
SEED         = 95
OUTPUT        = Path('hilo/output/regret.png')
ACTION_OUTPUT = Path('hilo/output/action_regret.png')
FIT_OUTPUT    = Path('hilo/output/gp_1d.png')   # written only when DIM == 1

# acquisition knobs, each read only by the strategy named
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

NUM_RESTARTS = 8        # L-BFGS-B starting points per acquisition optimization
RAW_SAMPLES  = 512      # Sobol samples scanned to pick them
SPREAD_REF   = 4096     # samples used to measure the truth's spread

ACQFS = {
    'ucb'    : partial(
        UpperConfidenceBound,
        beta = UCB_BETA
    ),
    'logei'  : LogExpectedImprovement,
    'lognei' : partial(
        LogNoisyExpectedImprovement,
        num_fantasies = NUM_FANTASIES
    ),
    'qlognei': partial(
        qLogNoisyExpectedImprovement,
        sampler        = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=SEED),
        prune_baseline = PRUNE_BASELINE
    )
}
STRATEGIES = ('random', *ACQFS)

PLOT_RES          = 400 # grid resolution of the 1D fit plot
POSTERIOR_SAMPLES = 20  # sample paths drawn on it
BAND_STD          = 2.0 # width of the shaded band, in standard deviations

# Okabe-Ito, a published colorblind-safe palette
SIMPLE_COLOR    = '#0072B2'
INFERENCE_COLOR = '#D55E00'

# a history row is (measurements, value simple, value inference, action simple,
# action inference, acquisition seconds), so a curve is one column of it
VALUE_CURVES = (
    (1, SIMPLE_COLOR,    'o', 'simple regret (best true value sampled)'),
    (2, INFERENCE_COLOR, 's', 'inference regret (true value at the posterior argmax)')
)
ACTION_CURVES = (
    (3, SIMPLE_COLOR,    'o', 'closest action sampled'),
    (4, INFERENCE_COLOR, 's', 'the posterior argmax itself')
)


def fit_gp(objective):
    """One GP on everything measured so far, hyperparameters refit."""
    return plr.BoTorchGP(objective, noise=GP_NOISE, fit_hyperparameters=True,
                         min_length_scale=MIN_LENGTHSCALE)


def acquisition(objective):
    """The `AcquisitionFunction` STRATEGY names, searching the declared box.
    """
    if STRATEGY not in ACQFS:
        raise ValueError(f'STRATEGY must be one of {STRATEGIES}, got {STRATEGY!r}')

    if STRATEGY == 'lognei' and plr.NoiseModel.coerce(GP_NOISE).is_fitted:
        raise ValueError("'lognei' needs a FixedNoiseGaussianLikelihood, so set "
                         'GP_NOISE = plr.NoiseModel.pinned(...) to use it')

    return plr.AcquisitionFunction(ACQFS[STRATEGY], objective,
                                   num_restarts=NUM_RESTARTS,
                                   raw_samples=RAW_SAMPLES)


def report(history, inferred, truth):
    """Both metrics and the acquisition cost at every budget, as one table.
    """
    print(f'\n=== {STRATEGY} on Levy, {DIM}D, noise {NOISE:.2f} of spread ===')
    print(f'  {"":>6}{"value regret":^23}{"action distance":^24}')
    print(f'  {"points":>6}  {"sampled":>8}  {"recommended":>11}'
          f'  {"sampled":>9}  {"recommended":>11}  {"acq (s)":>9}')
    for n, v_simple, v_inferred, a_simple, a_inferred, seconds in history:
        print(f'  {n:>6}  {v_simple:>8.4f}  {v_inferred:>11.4f}'
              f'  {a_simple:>9.4f}  {a_inferred:>11.4f}  {seconds:>9.3f}')

    print(f'\n  {"total acquisition":<22}{sum(row[-1] for row in history):.2f} s')
    print(f'  {"recommended action":<22}{np.round(inferred, 3)}')
    print(f'  {"true optimizers":<22}{np.round(truth.optimizers.numpy(), 3)}\n')


def make_figure(history, curves, ylabel, output):
    """The named curves against the number of measurements.

    Args:
        curves: (column, color, marker, label) per curve, indexing a history row.
        ylabel: what the two curves share, and so what the axis measures.
    """
    rows = np.array(history)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for column, color, marker, label in curves:
        ax.plot(rows[:, 0], rows[:, column], color=color, marker=marker, ms=4,
                label=label)

    ax.set_yscale('log')
    # a run spans well under a decade, so the major ticks alone label nothing
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis='y', which='minor', labelsize=7)

    ax.set_xlabel('measurements')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{STRATEGY} on Levy, {DIM}D')
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def make_fit_figure(truth, objective, gp, inferred, box, output):
    """The 1D fit: the GP and the truth evaluated on a grid, drawn by
    `plr.plot_fit_1d`.
    """
    grid    = np.linspace(box[0, 0], box[1, 0], PLOT_RES)[:, None]
    mu, std = gp.posterior_at(grid, raw=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plr.plot_fit_1d(
        ax       = ax,
        x        = grid[:, 0],
        mu       = mu[:, 0],
        std      = std[:, 0],
        xdata    = objective.xdata[:, 0],
        ydata    = objective.ydata,
        truth    = plr.truth_at(truth, grid),
        # one objective, so the single column is the path itself
        paths    = gp.sample_paths(grid, POSTERIOR_SAMPLES, raw=True)[:, :, 0],
        vlines   = {'recommended action': inferred[0],
                    'true optimizer'    : truth.optimizers[0, 0].item()},
        band_std = BAND_STD,
        title    = f'{STRATEGY} on Levy, 1D, after {len(objective.ydata)} measurements'
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def main():
    torch.manual_seed(SEED)

    box = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float)

    truth = plr.construct_function(
        func = plr.SYNTHETIC_FUNCTIONS['Levy'],
        dim  = DIM,
        box  = BOX,
        seed = SEED,
    )
    observe = plr.SyntheticFunction(
        truth         = truth, 
        rel_noise_std = NOISE,
        n_spread      = SPREAD_REF,
        seed          = SEED
    )

    objective = plr.Objective.from_empty(
        name          = 'Levy',
        maximize      = False,
        action_bounds = (-BOX, BOX)
    )

    opening = plr.sample_actions(box, 1, 'uniform', seed=SEED)
    objective.add_points(opening, observe(opening))

    acq       = acquisition(objective)
    recommend = plr.AcquisitionFunction(PosteriorMean, objective,
                                        num_restarts=NUM_RESTARTS,
                                        raw_samples=RAW_SAMPLES)

    history = []
    seconds = 0.0   # no acquisition chose the opening action
    for step in range(NUM_STEPS + 1):
        gp = fit_gp(objective)

        inferred = recommend.query(gp, q=1)[0]
        history.append((len(objective.ydata),
                        *plr.regret(truth, objective, inferred),
                        *plr.action_distance(truth, objective, inferred),
                        seconds))
        if step == NUM_STEPS:
            break

        start = time.perf_counter()
        action = acq.query(gp, q=1)[0]
        seconds = time.perf_counter() - start
        objective.add_points(action, observe(action[None]))

    report(history, inferred, truth)
    make_figure(history, VALUE_CURVES, 'regret', OUTPUT)
    make_figure(history, ACTION_CURVES,
                'distance to the nearest optimizer (box spans)', ACTION_OUTPUT)
    if DIM == 1:
        make_fit_figure(truth, objective, gp, inferred, box, FIT_OUTPUT)


if __name__ == '__main__':
    main()
