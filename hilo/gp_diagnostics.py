"""Leave-one-out cross-validation and R^2 for one GP on a synthetic objective.
"""

import argparse
from functools import partial
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
    SyntheticTestFunction
)
from botorch.utils.sampling import draw_sobol_samples
from sklearn.metrics import r2_score

import pypolar as plr
from pypolar.optimization.gp import DTYPE

DIM          = 1        # action dimension
NUM_SAMPLES  = 30       # measurements drawn from the objective
NOISE        = 0.710    # noise added, as a fraction of the truth's spread
GP_NOISE     = 'prior:0.3'  # a fraction, 'match', 'fit', or 'prior:MEDIAN'
MIN_LENGTHSCALE = None  # lengthscale floor, or None; scale it to the design
REFIT_FOLDS  = False    # refit kernel hyperparameters inside every LOO fold
FUNCTION     = 'levy'
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
DESIGN       = 'sobol'  # 'sobol' (space-filling) or 'uniform' (iid)
SEED         = 95
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


def make_truth(name: str, dim: int, box: float) -> SyntheticTestFunction:
    return FUNCTIONS[name](dim=dim, negate=True, bounds=[(-box, box)] * dim)


def truth_at(truth, X):
    with torch.no_grad():
        return truth(torch.as_tensor(X, dtype=DTYPE), noise=False).numpy()


def sample_design(bounds, n, kind, seed):
    """n actions over the box: Sobol is space-filling, uniform is iid."""
    if kind == 'sobol':
        return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1).numpy()

    lo, hi = bounds.numpy()
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, len(lo)))


def fit_gp(objective, noise, hypers=None, min_length_scale=None):
    """A GP on `objective`, with its hyperparameters refit or frozen.

    Args:
        objective: the measurements to condition on.
        noise: a `NoiseModel`, or a fraction of the objective's spread to pin,
            or None to fit it by marginal likelihood alongside the kernel.
        hypers: a `GPHyperparameters` to hold fixed, or None to fit them. Every
            field supersedes the arguments, the noise included, so a fitted
            noise freezes across folds exactly as the kernel does.
        min_length_scale: lower bound on every lengthscale, or None.
    """
    if hypers is None:
        return plr.BoTorchGP(objective, noise=noise, fit_hyperparameters=True,
                             min_length_scale=min_length_scale)

    # Standardize divides train_Y by its own sample spread, so the square root
    # of a post-Standardize noise variance is the fraction a pinned noise states
    return plr.BoTorchGP(
        objective,
        noise               = plr.NoiseModel.pinned(np.sqrt(hypers.noise_var)),
        fit_hyperparameters = False,
        length_scale        = hypers.lengthscale,
        signal_var          = hypers.signal_var,
        min_length_scale    = min_length_scale
    )


def calibration(residual, std):
    """Standard deviation of the z-scores and 95% interval coverage.

    Both read 1.0 and 0.95 when the posterior standard deviations are honest;
    a z-spread above 1 means the GP is overconfident, below 1 underconfident.
    """
    z = residual / std
    return z.std(), np.mean(np.abs(z) <= 1.96)


def report(args, y_obs, y_true, loo_mu, loo_std, in_sample_mu, noise_abs, gp_noise):
    """Everything the run measured, as one table."""
    folds = 'refit per fold' if args.refit_folds else 'frozen at the full-data fit'
    assumes = 'GP fitted' if plr.NoiseModel.coerce(args.gp_noise).is_fitted \
        else 'GP assumes'
    # no model can explain variance the noise put there, so R^2 against the
    # noisy observations cannot exceed this
    ceiling = 1 - noise_abs ** 2 / y_obs.var()

    print('\n=== setup ===')
    print(f'  {"function":<28}{args.function} on [{-args.box:g}, {args.box:g}]^{args.dim}, negated')
    print(f'  {"samples":<28}{args.samples} ({args.design})')
    print(f'  {"true noise":<28}{args.noise:.3f} of spread  =  {noise_abs:.4f}')
    print(f'  {assumes:<28}{gp_noise:.3f} of spread  =  {gp_noise * y_obs.std():.4f}')
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


def gp_noise_arg(value):
    """--gp-noise takes a fraction of the spread, 'match', 'fit', or 'prior:M'."""
    if value in ('match', 'fit') or value.startswith('prior:'):
        return value
    return float(value)


def resolve_gp_noise(value, true_noise):
    """The CLI spelling, as `fit_gp` takes it.

    'match' pins the GP at the oracle's own noise, which only a simulation can
    do; 'fit' leaves it free; 'prior:M' fits it under a LogNormal centered on M.
    """
    if value == 'match':
        return true_noise
    if value == 'fit':
        return None
    if isinstance(value, str):
        return plr.NoiseModel.prior(float(value.split(':', 1)[1]))
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dim',      type=int,   default=DIM,         help='action dimension')
    p.add_argument('--samples',  type=int,   default=NUM_SAMPLES, help='number of measurements')
    p.add_argument('--noise',    type=float, default=NOISE,       help="noise added, as a fraction of the truth's spread")
    p.add_argument('--gp-noise', type=gp_noise_arg, default=GP_NOISE,
                   help="noise the GP assumes: a fraction of spread, 'match' for "
                        "--noise, 'fit' to estimate it, or 'prior:M' to fit it "
                        "under a LogNormal centered on M")
    p.add_argument('--min-lengthscale', type=float, default=MIN_LENGTHSCALE,
                   help='floor every lengthscale, in the normalized action box')
    p.add_argument('--refit-folds', action='store_true', default=REFIT_FOLDS,
                   help='refit kernel hyperparameters inside every LOO fold')
    p.add_argument('--function', default=FUNCTION, choices=sorted(FUNCTIONS))
    p.add_argument('--box',      type=float, default=BOX,    help='the box is [-box, box]^dim')
    p.add_argument('--design',   default=DESIGN, choices=['sobol', 'uniform'])
    p.add_argument('--seed',     type=int,   default=SEED)
    p.add_argument('--output',   type=Path,  default=OUTPUT)

    args = p.parse_args()
    args.gp_noise = resolve_gp_noise(args.gp_noise, args.noise)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    truth  = make_truth(args.function, args.dim, args.box)
    bounds = truth.bounds

    X = draw_sobol_samples(
        bounds = bounds,
        n      = args.samples,
        q      = 1,
        seed   = args.seed
    ).squeeze(1) 
    noise_abs = args.noise * truth_at(truth, X).std() # noise is a fraction of the truth's own spread

    y_true = truth_at(truth, X)
    y_obs  = y_true + noise_abs * np.random.default_rng(args.seed).standard_normal(y_true.shape)

    objective = plr.Objective.from_data(
        actions  = X,
        values   = y_obs,
        maximize = True,
        name     = args.function
    )

    full_gp = fit_gp(objective, args.gp_noise, min_length_scale=args.min_lengthscale)
    hypers  = None if args.refit_folds else full_gp.get_fitted_hyperparameters()
    # a pinned noise reads back as exactly what it was given, so this one line
    # reports the effective level on either path
    gp_noise = float(np.sqrt(full_gp.get_fitted_hyperparameters().noise_var))

    # plr.loo calls fit_gp(objective, noise, hypers), so the floor is bound in
    loo_mu, loo_std, models = plr.loo(
        objective, partial(fit_gp, min_length_scale=args.min_lengthscale),
        args.gp_noise, hypers
    )
    in_sample_mu = full_gp.posterior_at(X, raw=True)[0][:, 0]

    report(
        args         = args,
        y_obs        = y_obs,
        y_true       = y_true,
        loo_mu       = loo_mu,
        loo_std      = loo_std,
        in_sample_mu = in_sample_mu,
        noise_abs    = noise_abs,
        gp_noise     = gp_noise
    )
    make_figure(args, truth, objective, models, full_gp, bounds.numpy(),
                y_true, loo_mu, loo_std)


if __name__ == '__main__':
    main()
