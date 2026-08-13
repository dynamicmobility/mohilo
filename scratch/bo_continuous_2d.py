"""Bayesian optimization over a continuous 2D action space.

Algorithm, per query, after a 2^3-point Sobol initial design:

    1. fit        mu(x), sigma(x) in closed form from the M observations
    2. incumbent  mu* = max mu over the observations
    3. stage 1    evaluate EI on 2^9 fresh scrambled Sobol points
    4. stage 2    box-constrained L-BFGS-B on EI from the best 8 -> x_next
    5. observe    y = f(x_next) + N(0, NOISE_STD^2), append

The recommendation at the end is the same two-stage maximization applied to mu
instead of EI. This is the scheme BoTorch's `optimize_acqf` implements: stage 1
only has to land a point inside the basin of the global EI maximum, which is
about one length scale wide, and stage 2 supplies the precision, at an accuracy
set by the solver tolerance rather than by the Sobol spacing.

Nothing here is discretized. `Regression` is unused because it stores feedback
by grid index and would snap every query to a lattice node; the GP is driven
directly, and the action space handed to `set_data` is just the M observations.
Everywhere else the posterior is read through `mu_at`/`std_at`, which evaluate
mu(x) = k(x, X_obs) G^-1 y and its variance at any x, on or off a grid.
`ExpectedImprovementSampler` is unused for the same reason: it scores a finite
list of actions, whereas EI is optimized here as a continuous function.

Run from the repo root:

    python scratch/bo_continuous_2d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

import pypolar as plr

LOW             = np.array([-2.0, -2.0])
HIGH            = np.array([ 2.0,  2.0])
BOUNDS          = list(zip(LOW, HIGH))
IDEAL_POINT     = np.array([ 0.63, -1.17])  # deliberately off any round lattice
REWARD_RANGE    = (-1.0, 1.0)               # BoundedIdealPoint output bounds
NOISE_STD       = 0.05
NUM_QUERIES     = 40
LOG2_INITIAL    = 3                         # 2^3 = 8 initial design points
LOG2_RAW        = 9                         # 2^9 = 512 stage-1 Sobol samples
NUM_RESTARTS    = 8                         # stage-2 L-BFGS-B starting points
XI              = 0.01                      # EI improvement threshold
CONTOUR_RES     = 200                       # plotting only
SEED            = 95


def sobol(log2_n, rng):
    """``2**log2_n`` scrambled Sobol points filling the box.

    Sobol's balance properties hold at powers of two, hence the log argument.
    Owen scrambling makes each point marginally uniform while preserving the low
    discrepancy of the set, and seeding the engine from ``rng`` advances it, so
    consecutive calls give independent sets.

    Args:
        log2_n: base-2 log of the number of points.
        rng: a numpy random Generator.

    Returns:
        (2**log2_n, d) array of points in [LOW, HIGH].
    """
    engine = qmc.Sobol(d=len(LOW), scramble=True, seed=rng)
    return qmc.scale(engine.random_base2(log2_n), LOW, HIGH)


def maximize(fn, rng):
    """Two-stage maximization of a cheap, differentiable field over the box.

    Stage 1 evaluates ``fn`` on a scrambled Sobol set; its only job is to place a
    point inside the basin of the global maximum. Stage 2 runs box-constrained
    L-BFGS-B from the best ``NUM_RESTARTS`` of them.

    Args:
        fn: maps an (n, d) array of actions to n values. Higher is better.
        rng: a numpy random Generator.

    Returns:
        The (d,) maximizer.
    """
    raw = sobol(LOG2_RAW, rng)
    starts = raw[np.argsort(fn(raw))[-NUM_RESTARTS:]]
    results = [
        minimize(lambda x: -fn(np.atleast_2d(x)).item(), x0=x0,
                 method='L-BFGS-B', bounds=BOUNDS)
        for x0 in starts
    ]
    return min(results, key=lambda res: res.fun).x


def expected_improvement(gp, X, incumbent):
    """EI over ``incumbent``, as a continuous function of the action.

        EI(x) = sigma(x) * (z Phi(z) + phi(z)),
        z     = (mu(x) - incumbent - XI) / sigma(x)

    The same closed form as ``ExpectedImprovementSampler.acquisition``, but read
    off ``mu_at``/``std_at`` rather than the stored ``mu``/``std()`` vectors, so
    it is defined and differentiable everywhere in the box.

    Args:
        gp: a fitted regression backend.
        X: an (n, d) array of actions.
        incumbent: best posterior mean among the observations.

    Returns:
        Length-n array of EI values.
    """
    mu, std = gp.mu_at(X), gp.std_at(X)
    safe = np.where(std > 0, std, 1.0)
    z = (mu - incumbent - XI) / safe
    return np.where(std > 0, std * (z * norm.cdf(z) + norm.pdf(z)), 0.0)


def plot(gp, X, x_star, groundtruth, path):
    """Ground truth, posterior mean and simple regret, side by side.

    Args:
        gp: the fitted GP.
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
        gp.mu_at(pts).reshape(X1.shape),
    ]

    # One level set shared by both panels, so the two are directly comparable.
    lo = min(field.min() for field in fields)
    hi = max(field.max() for field in fields)
    levels = np.linspace(lo, hi, 33)

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5), layout='constrained')
    titles = ['Ground truth reward', 'GP posterior mean (mu_at)']
    n_init = 2 ** LOG2_INITIAL

    for ax, field, title in zip(axs, fields, titles):
        im = ax.contourf(X1, X2, field, levels=levels, cmap='viridis')
        ax.scatter(*X[:n_init].T, marker='o', facecolor='none', edgecolor='w',
                   linewidth=1.2, label='Sobol initial design')
        ax.scatter(*X[n_init:].T, marker='o', color='w', edgecolor='k',
                   linewidth=0.5, s=32, label='EI queries')
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
    axs[2].annotate('EI takes over', xy=(n_init + 0.6, 0.96), fontsize=8,
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

    length_scale, signal_variance, precision = plr.derive_gp_hyperparams(
        domain_size    = float(np.max(HIGH - LOW)),
        expected_range = float(REWARD_RANGE[1] - REWARD_RANGE[0]),
        noise_var      = NOISE_STD ** 2,
    )

    groundtruth = plr.BoundedIdealPoint(
        w           = IDEAL_POINT,
        delta       = 0.6,
        gamma       = 0.0,
        lower_bound = REWARD_RANGE[0],
        upper_bound = REWARD_RANGE[1],
    )
    oracle = plr.NoisyRegressionOracle(groundtruth, NOISE_STD, rng)

    gp = plr.ConjugateGP(  # plr.BoTorchGP is a drop-in for the same contract
        kernel          = 'squared_exp',
        signal_variance = signal_variance,
        length_scale    = length_scale,
        rng             = rng,
    )

    X = sobol(LOG2_INITIAL, rng)
    y = np.array([oracle.query(x).item() for x in X])

    for _ in range(NUM_QUERIES - len(X)):
        # The action space given to the GP is the observations themselves; mu_at
        # and std_at cover every other point, so no candidate set is stored.
        gp.set_data(X, np.arange(len(X)), y, precision)
        gp.fit()

        # The incumbent is the best posterior mean rather than the best observed
        # value, which under noisy observations is biased upward.
        incumbent = gp.mu.max()
        action = maximize(lambda Z: expected_improvement(gp, Z, incumbent), rng)

        X = np.vstack([X, action])
        y = np.append(y, oracle.query(action).item())

    # Refit once more so the posterior reflects the final observation.
    gp.set_data(X, np.arange(len(X)), y, precision)
    gp.fit()

    x_star = maximize(gp.mu_at, rng)
    reported = [
        ('best queried action', X[np.argmax(gp.mu)]),
        ('argmax mu, off grid', x_star),
        ('true optimum',        IDEAL_POINT),
    ]
    for name, x in reported:
        print(f'{name:20s} a = [{x[0]: .4f}, {x[1]: .4f}]   '
              f'f(a) = {groundtruth(x).item(): .6f}')

    plot(gp, X, x_star, groundtruth,
         Path('scratch/output/bo_continuous_2d.png'))


if __name__ == '__main__':
    main()
