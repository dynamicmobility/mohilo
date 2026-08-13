"""`bo_continuous_2d.py`, with every step of the loop done by BoTorch.

The two scripts solve the identical problem — same 2D box, same ideal point,
same noise, same seed, same hyperparameter heuristic — so their outputs are
directly comparable. Only the machinery differs:

| step               | bo_continuous_2d.py            | here                          |
|--------------------|--------------------------------|-------------------------------|
| initial design     | `scipy.stats.qmc.Sobol`        | `draw_sobol_samples`          |
| posterior          | `ConjugateGP` + `mu_at/std_at` | `SingleTaskGP.posterior`      |
| acquisition        | hand-written EI closed form    | `LogExpectedImprovement`      |
| maximizing it      | `maximize()`: Sobol + L-BFGS-B | `optimize_acqf`               |
| recommendation     | `maximize(gp.mu_at)`           | `optimize_acqf(PosteriorMean)`|

`pypolar` is still used for the *problem*: the groundtruth reward, the noisy
oracle, and `derive_gp_hyperparams`. Nothing in the optimization loop is
pypolar's.

Three points where the translation is not literal:

1. **EI vs. LogEI.** `LogExpectedImprovement` returns log(EI) rather than EI.
   log is strictly increasing, so the maximizer is unchanged; the point is that
   EI underflows to exactly 0 in float64 several length scales from the data,
   which flattens the gradient L-BFGS-B needs. LogEI stays finite there.
2. **The `xi` threshold.** BoTorch's analytic EI has no `xi` argument, because
   EI over `incumbent + xi` is just EI with a raised reference. So `xi` is
   folded into `best_f`, which is exactly what the closed form in the other
   script does with `mu - incumbent - XI`.
3. **Choosing the L-BFGS-B starting points.** The other script takes the top
   `NUM_RESTARTS` of the `raw_samples` Sobol points. BoTorch's default
   initializer instead draws them by Boltzmann sampling on the acquisition
   values (temperature `eta`, default 1.0), so a mediocre-but-distant point can
   still be picked. That is deliberate: it spreads the restarts over several
   basins instead of clustering them all in the best one. Set `eta` large in
   `options` to recover top-k.

Run from the repo root:

    python scratch/bo_continuous_2d_botorch.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean

import pypolar as plr

LOW             = np.array([-2.0, -2.0])
HIGH            = np.array([ 2.0,  2.0])
IDEAL_POINT     = np.array([ 0.63, -1.17])  # deliberately off any round lattice
REWARD_RANGE    = (-1.0, 1.0)               # BoundedIdealPoint output bounds
NOISE_STD       = 0.05
NUM_QUERIES     = 40
LOG2_INITIAL    = 3                         # 2^3 = 8 initial design points
RAW_SAMPLES     = 512                       # stage-1 Sobol samples per query
NUM_RESTARTS    = 8                         # stage-2 L-BFGS-B starting points
XI              = 0.01                      # EI improvement threshold
CONTOUR_RES     = 200                       # plotting only
SEED            = 95
DTYPE           = torch.float64
BOUNDS          = torch.as_tensor(np.stack([LOW, HIGH]), dtype=DTYPE)  # (2, d), as optimize_acqf wants


def build_model(X, Y, sigma2, signal_variance, length_scale):
    """A fixed-hyperparameter exact GP over the observations so far.

    The configuration matches `BoTorchGP` and therefore `ConjugateGP`:

    - `train_Yvar` pins the observation noise, which selects
      `FixedNoiseGaussianLikelihood`, so nothing about the noise is inferred.
    - `ZeroMean`, matching the zero prior mean the other backends assume.
    - `outcome_transform=None` must be explicit: recent BoTorch defaults to
      `Standardize`, which would rescale `Y` out from under `signal_variance`.
    - `ScaleKernel(RBFKernel)` with `outputscale = signal_variance**2` and one
      shared lengthscale, i.e. k(a,b) = sv^2 exp(-||a-b||^2 / (2 ls^2)).

    No `fit_gpytorch_mll` call: the hyperparameters come from the same heuristic
    the other script uses, so the two posteriors are the same object. Fitting
    them by marginal likelihood is one added line, but it would make the
    comparison a comparison of two different models.

    Args:
        X: (n, d) observed actions.
        Y: (n,) observed values.
        sigma2: observation noise variance.
        signal_variance: kernel amplitude (squared to get the outputscale).
        length_scale: kernel lengthscale.

    Returns:
        A `SingleTaskGP` in eval mode.
    """
    train_X = torch.as_tensor(X, dtype=DTYPE)
    train_Y = torch.as_tensor(Y, dtype=DTYPE).reshape(-1, 1)

    covar = ScaleKernel(RBFKernel()).to(DTYPE)
    # Assign as float64 tensors, not Python floats: gpytorch inverts the Positive
    # constraint at the value's own dtype, and a Python float lands there as
    # float32, costing ~1e-8 on the lengthscale and ~3e-7 on the posterior mean.
    covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
    covar.outputscale = torch.tensor(signal_variance ** 2, dtype=DTYPE)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.full_like(train_Y, sigma2),
        covar_module=covar,
        mean_module=ZeroMean(),
        outcome_transform=None,
        input_transform=None,
    )
    return model.eval()


def maximize(acqf):
    """Two-stage maximization of an acquisition function over the box.

    `optimize_acqf` is the two-stage scheme itself: it evaluates `acqf` on
    `raw_samples` Sobol points, picks `num_restarts` starting points from them,
    and runs box-constrained L-BFGS-B on each, using gradients backpropagated
    through the GP rather than finite differences.

    Args:
        acqf: an `AcquisitionFunction`, scored on `(b, q, d)` batches.

    Returns:
        The (d,) maximizer, as numpy.
    """
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=BOUNDS,
        q=1,                            # one action per query, not a batch
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    return candidate.squeeze(0).detach().numpy()


def posterior_mean_at(model, X, chunk=4096):
    """Posterior mean at arbitrary points, as numpy.

    Each point is its own `q=1` batch element, so the posteriors are 1x1 and
    gpytorch never forms the n x n test-test covariance block — the same reason
    `BoTorchGP.mu_at` drops to data-space algebra. Chunking bounds peak memory
    on the plotting grid.

    Args:
        model: a fitted `SingleTaskGP`.
        X: (n, d) array of points.
        chunk: batch elements evaluated at once.

    Returns:
        Length-n array of posterior means.
    """
    acqf = PosteriorMean(model)
    X = torch.as_tensor(np.atleast_2d(X), dtype=DTYPE).unsqueeze(1)  # (n, 1, d)
    with torch.no_grad():
        out = [acqf(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)]
    return torch.cat(out).numpy()


def plot(model, X, x_star, groundtruth, path):
    """Ground truth, posterior mean and simple regret, side by side.

    Args:
        model: the fitted GP.
        X: (n, d) queried actions, initial design first.
        x_star: the continuous recommendation.
        groundtruth: the noiseless reward.
        path: where to write the figure.
    """
    axis = np.linspace(LOW, HIGH, CONTOUR_RES)
    X1, X2 = np.meshgrid(axis[:, 0], axis[:, 1])
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    fields = [
        groundtruth(pts).reshape(X1.shape),
        posterior_mean_at(model, pts).reshape(X1.shape),
    ]

    # One level set shared by both panels, so the two are directly comparable.
    lo = min(field.min() for field in fields)
    hi = max(field.max() for field in fields)
    levels = np.linspace(lo, hi, 33)

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5), layout='constrained')
    titles = ['Ground truth reward', 'GP posterior mean (BoTorch)']
    n_init = 2 ** LOG2_INITIAL

    for ax, field, title in zip(axs, fields, titles):
        im = ax.contourf(X1, X2, field, levels=levels, cmap='viridis')
        ax.scatter(*X[:n_init].T, marker='o', facecolor='none', edgecolor='w',
                   linewidth=1.2, label='Sobol initial design')
        ax.scatter(*X[n_init:].T, marker='o', color='w', edgecolor='k',
                   linewidth=0.5, s=32, label='LogEI queries')
        ax.scatter(*x_star, marker='*', color='w', edgecolor='k', linewidth=0.6,
                   s=320, zorder=3, label='Recommendation')
        ax.scatter(*IDEAL_POINT, marker='X', color='w', edgecolor='k',
                   linewidth=0.6, s=130, zorder=3, label='True optimum')
        ax.set_title(title)
        ax.set_xlabel('$a_1$')
        ax.set_ylabel('$a_2$')
        ax.set_aspect('equal')

    axs[0].legend(loc='upper left', framealpha=0.9, fontsize=8)
    fig.colorbar(im, ax=axs[:2], label='Reward', shrink=0.85)

    # Simple regret: how far the best action queried so far is from the optimum,
    # measured in noiseless reward, so the oracle noise does not flatter it.
    best_seen = np.maximum.accumulate(groundtruth(X).ravel())
    regret = np.clip(groundtruth(IDEAL_POINT).item() - best_seen, 1e-12, None)
    axs[2].plot(np.arange(1, len(regret) + 1), regret, linewidth=2, color='k')
    axs[2].axvline(n_init, color='grey', linestyle='--', linewidth=1)
    axs[2].annotate('LogEI takes over', xy=(n_init + 0.6, 0.96), fontsize=8,
                    xycoords=('data', 'axes fraction'), color='grey',
                    ha='left', va='top')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Query')
    axs[2].set_ylabel('Simple regret')
    axs[2].set_title('Best queried action vs. the optimum')
    axs[2].grid(alpha=0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f'Saved figure to {path.resolve()}')


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)  # optimize_acqf's Sobol draws come from torch

    length_scale, signal_variance, precision = plr.derive_gp_hyperparams(
        domain_size    = float(np.max(HIGH - LOW)),
        expected_range = float(REWARD_RANGE[1] - REWARD_RANGE[0]),
        noise_var      = NOISE_STD ** 2,
    )
    # pypolar states the likelihood as `precision * ||Sr - y||^2`, a Gaussian
    # NLL with sigma^2 = 1/(2*precision). BoTorch wants the variance directly.
    sigma2 = 1.0 / (2.0 * precision)

    groundtruth = plr.BoundedIdealPoint(
        w           = IDEAL_POINT,
        delta       = 0.6,
        gamma       = 0.0,
        lower_bound = REWARD_RANGE[0],
        upper_bound = REWARD_RANGE[1],
    )
    oracle = plr.NoisyRegressionOracle(groundtruth, NOISE_STD, rng)

    X = draw_sobol_samples(
        BOUNDS, n=2 ** LOG2_INITIAL, q=1, seed=SEED
    ).squeeze(1).numpy()
    y = np.array([oracle.query(x).item() for x in X])

    for _ in range(NUM_QUERIES - len(X)):
        model = build_model(X, y, sigma2, signal_variance, length_scale)

        incumbent = posterior_mean_at(model, X).max()
        action = maximize(LogExpectedImprovement(model, best_f=incumbent + XI))

        X = np.vstack([X, action])
        y = np.append(y, oracle.query(action).item())

    # Refit once more so the posterior reflects the final observation.
    model = build_model(X, y, sigma2, signal_variance, length_scale)

    x_star = maximize(PosteriorMean(model))
    reported = [
        ('best queried action', X[np.argmax(posterior_mean_at(model, X))]),
        ('argmax mu, off grid', x_star),
        ('true optimum',        IDEAL_POINT),
    ]
    for name, x in reported:
        print(f'{name:20s} a = [{x[0]: .4f}, {x[1]: .4f}]   '
              f'f(a) = {groundtruth(x).item(): .6f}')

    plot(model, X, x_star, groundtruth,
         Path('scratch/output/bo_continuous_2d_botorch.png'))


if __name__ == '__main__':
    main()
