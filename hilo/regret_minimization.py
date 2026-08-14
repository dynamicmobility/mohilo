"""Sequential Bayesian optimization on a synthetic objective: simple and
inference regret against the strategy that chose the points.
"""

import time
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
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import (
    Ackley,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
    SyntheticTestFunction
)

import pypolar as plr
from pypolar.optimization.gp import DTYPE

DIM          = 1        # action dimension
NUM_INIT     = 5        # points in the initial design, before any model
NUM_STEPS    = 20       # acquisition steps taken after it
NOISE        = 0.5      # noise added, as a fraction of the truth's spread
GP_NOISE     = plr.NoiseModel.prior(0.5)  # a float pins, None fits, prior() regularizes
MIN_LENGTHSCALE = 0.1   # lengthscale floor, at the design spacing
STRATEGIES   = ('random', 'ucb', 'logei', 'lognei', 'qlognei')
STRATEGY     = 'logei'  # one of STRATEGIES; 'random' is the non-adaptive baseline
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
SEED         = 95
OUTPUT       = Path('hilo/output/regret.png')
FIT_OUTPUT   = Path('hilo/output/gp_1d.png')   # written only when DIM == 1

# acquisition knobs, each read only by the strategy named
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

NUM_RESTARTS = 8        # L-BFGS-B starting points per acquisition optimization
RAW_SAMPLES  = 512      # Sobol samples scanned to pick them
SPREAD_REF   = 4096     # samples used to measure the truth's spread

PLOT_RES          = 400 # grid resolution of the 1D fit plot
POSTERIOR_SAMPLES = 20  # sample paths drawn on it
BAND_STD          = 2.0 # width of the shaded band, in standard deviations

# Okabe-Ito, a published colorblind-safe palette
SIMPLE_COLOR    = '#0072B2'
INFERENCE_COLOR = '#D55E00'
INIT_COLOR      = '#56B4E9'
TRUTH_COLOR     = '#000000'
SAMPLE_COLOR    = '#009E73'


def make_truth(dim, box):
    """Levy on [-box, box]^dim.

    Not negated: `Objective(maximize=False)` is what encodes the direction, and
    negating here too would double-flip it.
    """
    return Ackley(dim=dim, bounds=[(-box, box)] * dim)


def truth_at(truth, X):
    """Noiseless values of the truth at the (n, d) actions X."""
    with torch.no_grad():
        return truth(torch.as_tensor(X, dtype=DTYPE), noise=False).numpy()


def noise_scale(truth, bounds, seed):
    """The absolute standard deviation NOISE names as a fraction of the truth's
    spread, measured once over the whole box.

    Measured once rather than per batch, because a sequential loop adds one
    point at a time and a single point has no spread of its own to take a
    fraction of.
    """
    X = plr.sample_actions(bounds=bounds, n=SPREAD_REF, kind='sobol', seed=seed)
    return NOISE * truth_at(truth, X).std()


def observe(truth, X, noise_abs, rng):
    """Noisy measurements at the (n, d) actions X, returned (n,)."""
    y = truth_at(truth, X)
    return y + noise_abs * rng.standard_normal(y.shape)


def fit_gp(objective):
    """One GP on everything measured so far, hyperparameters refit."""
    return plr.BoTorchGP(objective, noise=GP_NOISE, fit_hyperparameters=True,
                         min_length_scale=MIN_LENGTHSCALE)


def acquisition(gp, objective):
    """The acquisition function STRATEGY names.

    All of them read the posterior in `standard_y` units -- larger-is-better
    and standardized -- since `Standardize` is undone inside `model.posterior`,
    so a minimized objective needs no sign handling: it was negated on the way in.

    The difference between the plain and the noisy variants is the incumbent
    they improve on. LogEI takes `standard_y.max()`, the best value actually
    *observed*, which under noise is whichever point drew the luckiest draw --
    so it chases noise, and the more it samples the worse the bias gets. The
    noisy variants instead integrate over the posterior at the measured points,
    making the incumbent the model's belief about the best *true* value.
    """
    if STRATEGY == 'ucb':
        return UpperConfidenceBound(gp.model, beta=UCB_BETA)

    if STRATEGY == 'logei':
        return LogExpectedImprovement(gp.model, best_f=objective.standard_y.max())

    # the measured actions in the model's own frame, which is the normalized
    # one the GP was trained on rather than raw units
    X = torch.as_tensor(objective.normalized_x, dtype=DTYPE)

    if STRATEGY == 'lognei':
        # the incumbent comes from fantasizing noiseless values at the measured
        # points, which botorch only supports when the noise is known
        if plr.NoiseModel.coerce(GP_NOISE).is_fitted:
            raise ValueError("'lognei' needs a FixedNoiseGaussianLikelihood, so "
                             'set GP_NOISE = plr.NoiseModel.pinned(...) to use it')
        return LogNoisyExpectedImprovement(gp.model, X_observed=X,
                                           num_fantasies=NUM_FANTASIES)

    if STRATEGY == 'qlognei':
        return qLogNoisyExpectedImprovement(
            gp.model,
            X_baseline     = X,
            sampler        = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=SEED),
            prune_baseline = PRUNE_BASELINE
        )

    raise ValueError(f'STRATEGY must be one of {STRATEGIES}, got {STRATEGY!r}')


def maximize_over_box(acq, objective, box):
    """The action maximizing `acq`, over the whole action box.

    `BoTorchGP` normalizes actions by the range of the points measured so far,
    so the model's [0, 1]^d is the bounding box of the data rather than the box
    the experiment may sample. Mapping `box` through that same transform is
    what lets a proposal step outside the points already collected; optimizing
    over [0, 1]^d would trap the loop inside its own convex hull.
    """
    bounds = torch.as_tensor(objective.xtransform(box), dtype=DTYPE)
    candidate, _ = optimize_acqf(
        acq_function = acq,
        bounds       = bounds,
        q            = 1,
        num_restarts = NUM_RESTARTS,
        raw_samples  = RAW_SAMPLES
    )

    # the round trip through the transform can land a hair outside the box,
    # which the test function rejects
    action = objective.xtransform.inv(candidate.detach().numpy())[0]
    return np.clip(action, box[0], box[1])


def regret(truth, objective, inferred):
    """Simple and inference regret, both against the true optimum.

    Simple regret is the best *noiseless* value sampled so far, so a lucky draw
    of the noise cannot flatter it. Inference regret is the true value at the
    action the run would recommend right now -- the posterior argmax -- which
    is what a human-in-the-loop study hands back to a subject.
    """
    return (
        truth_at(truth, objective.xdata).min() - truth.optimal_value,
        truth_at(truth, inferred[None]).item() - truth.optimal_value
    )


def report(history, inferred, truth):
    """Regret and acquisition cost at every budget, as one table.

    The seconds are the acquisition that chose that row's newest point, so the
    initial design reads zero and the cost knobs show up here directly.
    """
    print(f'\n=== {STRATEGY} on Levy, {DIM}D, noise {NOISE:.2f} of spread ===')
    print(f'  {"points":>6}  {"simple regret":>14}  {"inference regret":>17}  {"acq (s)":>9}')
    for n, simple, inference, seconds in history:
        print(f'  {n:>6}  {simple:>14.4f}  {inference:>17.4f}  {seconds:>9.3f}')

    print(f'\n  {"total acquisition":<22}{sum(row[3] for row in history):.2f} s')
    print(f'  {"recommended action":<22}{np.round(inferred, 3)}')
    print(f'  {"true optimizer":<22}{np.round(truth.optimizers[0].numpy(), 3)}\n')


def make_figure(history, output):
    """Both regret curves against the number of measurements.

    Log scale, because a working strategy drives regret down by orders of
    magnitude and a linear axis hides everything after the first few steps.
    """
    n, simple, inference, _ = np.array(history).T

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(n, simple, color=SIMPLE_COLOR, marker='o', ms=4,
            label='simple regret (best true value sampled)')
    ax.plot(n, inference, color=INFERENCE_COLOR, marker='s', ms=4,
            label='inference regret (true value at the posterior argmax)')
    ax.axvline(NUM_INIT, color=INIT_COLOR, ls='--', lw=1,
               label='end of the initial design')

    ax.set_yscale('log')
    # a run spans well under a decade, so the major ticks alone label nothing
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis='y', which='minor', labelsize=7)

    ax.set_xlabel('measurements')
    ax.set_ylabel('regret')
    ax.set_title(f'{STRATEGY} on Levy, {DIM}D')
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def posterior_paths(gp, objective, grid):
    """`POSTERIOR_SAMPLES` sample paths of the GP over `grid`, in raw units.

    Drawn from the *joint* posterior over the whole grid, which is why this
    does not go through `posterior_at`: that feeds its actions as q=1 batch
    elements so gpytorch never forms the n x n test-test block, which makes it
    cheap but leaves it marginal. Sampling each grid point from its own
    marginal independently would draw white noise rather than a function.
    """
    X = torch.as_tensor(objective.xtransform(grid), dtype=DTYPE)
    with torch.no_grad():
        draws = gp.model.posterior(X).rsample(torch.Size([POSTERIOR_SAMPLES]))

    # the posterior reports standardized larger-is-better values, so the paths
    # go back through the same inverse the mean does
    return objective.to_raw(draws.squeeze(-1).numpy())


def plot_fit_1d(truth, objective, gp, inferred, box, output):
    """The 1D fit: the truth, the GP's mean and band, sample paths from its
    posterior, and the measurements it was fit to.

    The band is the *latent* posterior, so it is the GP's uncertainty about the
    noiseless function -- what the truth curve should fall inside. It is not a
    predictive interval for a new measurement, which would be wider by the
    observation noise.
    """
    grid    = np.linspace(box[0, 0], box[1, 0], PLOT_RES)[:, None]
    mu, std = gp.posterior_at(grid, raw=True)
    mu, std = mu[:, 0], std[:, 0]
    xs      = grid[:, 0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, path in enumerate(posterior_paths(gp, objective, grid)):
        ax.plot(xs, path, color=SAMPLE_COLOR, lw=0.7, alpha=0.35, zorder=1,
                label=f'{POSTERIOR_SAMPLES} posterior samples' if i == 0 else None)

    ax.fill_between(xs, mu - BAND_STD * std, mu + BAND_STD * std,
                    color=INFERENCE_COLOR, alpha=0.18, zorder=2,
                    label=f'GP mean $\\pm$ {BAND_STD:g}$\\sigma$')
    ax.plot(xs, mu, color=INFERENCE_COLOR, lw=1.8, ls='--', zorder=3, label='GP mean')
    ax.plot(xs, truth_at(truth, grid), color=TRUTH_COLOR, lw=1.8, zorder=4, label='truth')
    ax.scatter(objective.xdata[:, 0], objective.ydata, s=22, color=TRUTH_COLOR,
               zorder=6, label='measurements')

    ax.axvline(inferred[0], color=INFERENCE_COLOR, ls='--', lw=1.2, zorder=5,
               label='recommended action')
    ax.axvline(truth.optimizers[0, 0].item(), color=TRUTH_COLOR, ls=':', lw=1.2,
               zorder=5, label='true optimizer')

    ax.set_xlabel('action')
    ax.set_ylabel('objective')
    ax.set_title(f'{STRATEGY} on Levy, 1D, after {len(objective.ydata)} measurements')
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    truth     = make_truth(DIM, BOX)
    box       = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float)
    bounds    = torch.as_tensor(box, dtype=DTYPE)
    noise_abs = noise_scale(truth, bounds, SEED)

    # every strategy takes its initial design, and 'random' its whole design,
    # from this one sequence, so a regret difference is the acquisition's doing
    # rather than a luckier initialization
    pool = plr.sample_actions(bounds=bounds, n=NUM_INIT + NUM_STEPS,
                              kind='sobol', seed=SEED)

    objective = plr.Objective.from_empty('Levy', maximize=False)
    objective.add_points(pool[:NUM_INIT],
                         observe(truth, pool[:NUM_INIT], noise_abs, rng))

    history = []
    seconds = 0.0   # no acquisition chose the initial design
    for step in range(NUM_STEPS + 1):
        gp = fit_gp(objective)

        # the posterior argmax over the same box the acquisition searches, not
        # BoTorchGP.best_actions, which searches [0, 1]^d -- the data's own
        # bounding box -- and so could not name a recommendation outside it
        inferred = maximize_over_box(PosteriorMean(gp.model), objective, box)
        history.append((len(objective.ydata), *regret(truth, objective, inferred), seconds))
        if step == NUM_STEPS:
            break

        # construction and optimization together: prune_baseline does its work
        # when the acquisition is built, not when it is evaluated
        start = time.perf_counter()
        action = (pool[NUM_INIT + step] if STRATEGY == 'random'
                  else maximize_over_box(acquisition(gp, objective), objective, box))
        seconds = time.perf_counter() - start

        objective.add_points(action, observe(truth, action[None], noise_abs, rng))

    report(history, inferred, truth)
    make_figure(history, OUTPUT)
    if DIM == 1:
        # gp and inferred are the last loop iteration's, so both are the
        # full-data fit that produced the recommendation just reported
        plot_fit_1d(truth, objective, gp, inferred, box, FIT_OUTPUT)


if __name__ == '__main__':
    main()
