"""Leave-one-out cross-validation and R^2 for one GP on a synthetic objective.

The point is to answer, on data whose ground truth is known exactly, two
questions a fit to real pilot data cannot answer:

1. **Does the GP predict measurements it has not seen?** That is leave-one-out:
   for each of the N measurements, refit on the other N-1 and predict the one
   that was held out. Nothing in fold i has seen point i -- not the kernel, not
   the standardization, not the action normalization, since every fold builds
   its own `Objective`. R^2 over those held-out predictions is the honest
   version of R^2; the in-sample R^2 printed beside it is the same GP scored on
   its own training points, and the gap between them is how much the fit is
   memorizing rather than generalizing.

2. **Are its error bars honest?** A GP reports a standard deviation, so the
   held-out residuals can be divided by it. If the posterior is calibrated those
   z-scores are standard normal, so their standard deviation is 1 and 95% of the
   truth falls inside the 95% interval. R^2 alone cannot see this: a model can
   predict well and still be systematically over- or under-confident.

Because the objective is synthetic, both are reported twice -- against the noisy
measurements (what you could compute from real data) and against the noiseless
truth (what you actually want to know). The first is capped by the noise: no
model can explain variance that is not there, so its ceiling is printed too.

    conda activate pypolar
    python -m hilo.gp_diagnostics --dim 1 --samples 40 --noise 0.05

Two noise levels are deliberately separate. `--noise` is the noise actually added
to the measurements; `--gp-noise` is what the GP is told to assume. Both are
fractions of a spread, matching `BoTorchGP`'s own convention, so equal values
mean a well-specified GP and unequal ones a deliberately misspecified one.

Needs `matplotlib` and `scikit-learn`, neither of which is a package dependency
-- both are experiment-only imports, as `pandas` is in plot_pilot.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions import (
    Ackley,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
)
from botorch.utils.sampling import draw_sobol_samples
from sklearn.metrics import r2_score

from pypolar.optimization.gp import DTYPE, BoTorchGP
from pypolar.optimization.objectives import Objective

DIM          = 1        # action dimension
NUM_SAMPLES  = 15       # measurements drawn from the objective
NOISE        = 0.25     # noise added, as a fraction of the truth's spread
GP_NOISE     = None     # noise the GP assumes; None matches NOISE
REFIT_FOLDS  = False    # refit kernel hyperparameters inside every LOO fold
FUNCTION     = 'levy'
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
DESIGN       = 'sobol'  # 'sobol' (space-filling) or 'uniform' (iid)
SEED         = 0
OUTPUT       = Path('hilo/output/gp_diagnostics.png')

PLOT_RES     = 400      # plotting only
SPREAD_REF   = 4096     # samples used to measure the truth's spread

FUNCTIONS = {
    'ackley':     Ackley,
    'griewank':   Griewank,
    'levy':       Levy,
    'rastrigin':  Rastrigin,
    'rosenbrock': Rosenbrock,
    'styblinski': StyblinskiTang,
}

# Okabe-Ito, a published colorblind-safe palette
TRUTH_COLOR = '#000000'
FOLD_COLOR  = '#56B4E9'
FULL_COLOR  = '#D55E00'
POINT_COLOR = '#0072B2'


def make_truth(name, dim, box):
    """The synthetic objective over [-box, box]^dim, negated so that it is
    maximized. The default box matters: Ackley's own bounds are +/-32.77, where
    the function is a flat plateau almost everywhere and nothing is learnable."""
    return FUNCTIONS[name](dim=dim, negate=True, bounds=[(-box, box)] * dim)


def truth_at(truth, X):
    """Noiseless values of the objective at X, as an (n,) array.

    `evaluate_true` does not apply `negate`, so it is the wrong way to ask for
    the ground truth of a negated function; calling the function with
    `noise=False` applies the negation and skips only the noise.
    """
    with torch.no_grad():
        return truth(torch.as_tensor(X, dtype=DTYPE), noise=False).numpy()


def sample_design(bounds, n, kind, seed):
    """n actions over the box: Sobol is space-filling, uniform is iid."""
    if kind == 'sobol':
        return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1).numpy()

    lo, hi = bounds.numpy()
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, len(lo)))


def fit_gp(objective, noise_std, hypers=None):
    """A GP on `objective`, with kernel hyperparameters refit or frozen.

    Args:
        objective: the measurements to condition on.
        noise_std: observation noise, as a fraction of the objective's spread.
        hypers: `(lengthscale, signal_var)` to hold fixed, or None to fit them.
    """
    gp = BoTorchGP(objective, noise_std=noise_std, fit_hyperparameters=hypers is None)
    if hypers is not None:
        # BoTorchGP reads these when it builds, so the model constructed in
        # __init__ is thrown away and rebuilt on the frozen values
        gp.LENGTH_SCALE, gp.SIGNAL_VAR = hypers
        gp.update_feedback(objective)

    return gp


def hyperparameters(gp):
    """The fitted ARD lengthscales and signal variance of a fitted GP."""
    kernel = gp.model.covar_module
    return (kernel.base_kernel.lengthscale.detach().numpy().ravel(),
            kernel.outputscale.item())


def leave_one_out(objective, noise_std, hypers):
    """Held-out posterior at every measured action.

    Fold i is refit on the other N-1 measurements and asked to predict point i.
    The fold builds its own `Objective`, so its standardization and its action
    normalization are computed without the held-out point too -- leaking the
    mean of a point into the transform that predicts it would flatter the
    result, especially at small N.

    Args:
        objective: the full set of measurements.
        noise_std: what each fold's GP assumes, a fraction of its own spread.
        hypers: `(lengthscale, signal_var)` frozen in every fold, or None to
            refit them fold by fold.

    Returns:
        (mu, std, models): length-N held-out mean and standard deviation in raw
        units, and the GP of each fold.
    """
    n = objective.ydata.size
    mu, std, models = np.empty(n), np.empty(n), []

    for i in range(n):
        keep = np.delete(np.arange(n), i)
        fold = Objective.from_data(
            actions  = objective.xdata[keep],
            values   = objective.ydata[keep],
            maximize = objective.maximize,
            name     = objective.name
        )
        gp = fit_gp(fold, noise_std, hypers)
        fold_mu, fold_std = gp.posterior_at(objective.xdata[i], raw=True)

        mu[i], std[i] = fold_mu[0, 0], fold_std[0, 0]
        models.append(gp)

    return mu, std, models


def calibration(residual, std):
    """Standard deviation of the z-scores and 95% interval coverage.

    Both read 1.0 and 0.95 when the posterior standard deviations are honest;
    a z-spread above 1 means the GP is overconfident, below 1 underconfident.
    """
    z = residual / std
    return z.std(), np.mean(np.abs(z) <= 1.96)


def report(args, y_obs, y_true, loo_mu, loo_std, in_sample_mu, noise_abs, gp_noise_abs):
    """Everything the run measured, as one table."""
    folds = 'refit per fold' if args.refit_folds else 'frozen at the full-data fit'
    # no model can explain variance the noise put there, so R^2 against the
    # noisy observations cannot exceed this
    ceiling = 1 - noise_abs ** 2 / y_obs.var()

    print('\n=== setup ===')
    print(f'  {"function":<28}{args.function} on [{-args.box:g}, {args.box:g}]^{args.dim}, negated')
    print(f'  {"samples":<28}{args.samples} ({args.design})')
    print(f'  {"true noise":<28}{args.noise:.3f} of spread  =  {noise_abs:.4f}')
    print(f'  {"GP assumes":<28}{args.gp_noise:.3f} of spread  =  {gp_noise_abs:.4f}')
    print(f'  {"fold hyperparameters":<28}{folds}')

    print('\n=== R^2 ===')
    print(f'  {"LOO vs observations":<28}{r2_score(y_obs, loo_mu):>8.4f}   (ceiling {ceiling:.4f}, noise-limited)')
    print(f'  {"LOO vs noiseless truth":<28}{r2_score(y_true, loo_mu):>8.4f}')
    print(f'  {"in-sample vs observations":<28}{r2_score(y_obs, in_sample_mu):>8.4f}   (same GP on its own training points)')

    print('\n=== held-out error ===')
    rmse = np.sqrt(np.mean((y_true - loo_mu) ** 2))
    print(f'  {"RMSE vs noiseless truth":<28}{rmse:>8.4f}')
    print(f'  {"as a fraction of spread":<28}{rmse / y_true.std():>8.4f}')

    print('\n=== calibration vs noiseless truth ===')
    z_spread, coverage = calibration(y_true - loo_mu, loo_std)
    print(f'  {"z-score spread":<28}{z_spread:>8.4f}   (1.0 if the error bars are honest)')
    print(f'  {"95% interval coverage":<28}{coverage:>8.4f}   (0.95 if the error bars are honest)')
    print()


def plot_folds(ax, truth, objective, models, full_gp, bounds):
    """Every fold's posterior mean on one axes, over the truth and the samples.

    The spread between the faint curves is the diagnostic: it is how much the
    fit moves when any single measurement is removed, so a wide bundle means the
    fit is resting on individual points rather than on the design as a whole.
    """
    grid = np.linspace(bounds[0, 0], bounds[1, 0], PLOT_RES)[:, None]

    for i, gp in enumerate(models):
        ax.plot(grid[:, 0], gp.posterior_at(grid, raw=True)[0][:, 0],
                color=FOLD_COLOR, lw=0.8, alpha=0.35, zorder=1,
                label='LOO fold means' if i == 0 else None)

    ax.plot(grid[:, 0], full_gp.posterior_at(grid, raw=True)[0][:, 0],
            color=FULL_COLOR, lw=1.8, ls='--', zorder=3, label='full-data GP')
    ax.plot(grid[:, 0], truth_at(truth, grid), color=TRUTH_COLOR, lw=1.8,
            zorder=2, label='truth')
    ax.scatter(objective.xdata[:, 0], objective.ydata, s=18, color=TRUTH_COLOR,
               zorder=4, label='measurements')

    ax.set_xlabel('action')
    ax.set_ylabel('objective')
    ax.set_title('each fold refit without one point')


def plot_parity(ax, y_true, mu, std):
    """Held-out prediction against the truth it was hiding.

    The identity line is where a perfect model would put every point, so
    vertical distance from it is the residual R^2 is computed from. Error bars
    are the latent posterior, which is what pairs with the noiseless truth.
    """
    span = [min(y_true.min(), mu.min()), max(y_true.max(), mu.max())]
    ax.plot(span, span, color='0.6', ls='--', lw=1, zorder=1, label='perfect prediction')
    ax.errorbar(y_true, mu, yerr=2 * std, fmt='o', ms=4, color=POINT_COLOR,
                ecolor=POINT_COLOR, elinewidth=0.8, alpha=0.8, capsize=0,
                zorder=2, label='LOO prediction +/- 2sd')

    ax.set_xlabel('noiseless truth')
    ax.set_ylabel('held-out prediction')
    ax.set_title(f'leave-one-out, $R^2$ = {r2_score(y_true, mu):.3f}')


def make_figure(args, truth, objective, models, full_gp, bounds, y_true, mu, std):
    """The fold overlay (1D only, where a fit can be drawn) beside the parity
    plot, which reads the same in any action dimension."""
    one_d = args.dim == 1
    fig, axes = plt.subplots(1, 2 if one_d else 1, figsize=(11 if one_d else 5.5, 4.5))
    axes = np.atleast_1d(axes)

    if one_d:
        plot_folds(axes[0], truth, objective, models, full_gp, bounds)

    plot_parity(axes[-1], y_true, mu, std)
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f'wrote {args.output}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dim',      type=int,   default=DIM,         help='action dimension')
    p.add_argument('--samples',  type=int,   default=NUM_SAMPLES, help='number of measurements')
    p.add_argument('--noise',    type=float, default=NOISE,       help="noise added, as a fraction of the truth's spread")
    p.add_argument('--gp-noise', type=float, default=GP_NOISE,    help='noise the GP assumes; defaults to --noise')
    p.add_argument('--refit-folds', action='store_true', default=REFIT_FOLDS,
                   help='refit kernel hyperparameters inside every LOO fold')
    p.add_argument('--function', default=FUNCTION, choices=sorted(FUNCTIONS))
    p.add_argument('--box',      type=float, default=BOX,    help='the box is [-box, box]^dim')
    p.add_argument('--design',   default=DESIGN, choices=['sobol', 'uniform'])
    p.add_argument('--seed',     type=int,   default=SEED)
    p.add_argument('--output',   type=Path,  default=OUTPUT)

    args = p.parse_args()
    args.gp_noise = args.noise if args.gp_noise is None else args.gp_noise
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    truth  = make_truth(args.function, args.dim, args.box)
    bounds = truth.bounds

    # the noise is a fraction of the truth's own spread, so that it is on the
    # same footing as the GP's noise_std and the two are directly comparable
    noise_abs = args.noise * truth_at(
        truth, draw_sobol_samples(bounds=bounds, n=SPREAD_REF, q=1, seed=args.seed).squeeze(1)
    ).std()

    X = sample_design(bounds, args.samples, args.design, args.seed)
    y_true = truth_at(truth, X)
    y_obs  = y_true + noise_abs * np.random.default_rng(args.seed).standard_normal(y_true.shape)

    objective = Objective.from_data(actions=X, values=y_obs, maximize=True,
                                    name=args.function)

    # the full-data fit does double duty: it is the in-sample baseline, and its
    # hyperparameters are what the folds freeze unless asked to refit
    full_gp = fit_gp(objective, args.gp_noise)
    hypers  = None if args.refit_folds else hyperparameters(full_gp)

    mu, std, models = leave_one_out(objective, args.gp_noise, hypers)
    in_sample_mu = full_gp.posterior_at(X, raw=True)[0][:, 0]

    report(args, y_obs, y_true, mu, std, in_sample_mu,
           noise_abs, args.gp_noise * y_obs.std())
    make_figure(args, truth, objective, models, full_gp, bounds.numpy(),
                y_true, mu, std)


if __name__ == '__main__':
    main()
