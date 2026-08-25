"""Leave-one-out cross-validation of a GP against its own measurements."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pypolar.optimization.objectives import Objective

MIN_SIZE = 5    # smallest subset `loo_curve` scores by default


def loo(objective: Objective, fit_gp, noise, hypers=None):
    """Held-out posterior at every measured action.

    Fold i is fit on every measurement except i, and is then asked to predict
    point i. No fold ever sees the point it is scored against, so the residuals
    are out-of-sample errors rather than the training fit.

    Every fold inherits `objective.action_bounds`, so a pinned action frame
    holds still across the folds. Without one the frame is each fold's own
    measured box, which moves whenever the dropped point was an extreme.

    Args:
        objective: the full set of measurements.
        fit_gp: builds one fold's fitted GP, called as
            `fit_gp(objective, noise, hypers)`. Everything a fold's GP needs
            beyond those three -- a lengthscale floor, say -- belongs bound into
            this callable, so the same configuration reaches every fold.
        noise: passed through to `fit_gp`; a `NoiseModel` or a shorthand it
            coerces.
        hypers: a `GPHyperparameters` for `fit_gp` to freeze in every fold, or
            None to refit them fold by fold.

    Returns:
        (mu, std, models): mu and std are the predicted (held-out) point for each
        respective fold in raw units. Models is the GP of each fold.
    """
    n = objective.ydata.size
    mu, std, models = np.empty(n), np.empty(n), []

    for i in range(n):
        keep = np.delete(np.arange(n), i)
        fold = Objective.from_data(
            actions       = objective.xdata[keep],
            values        = objective.ydata[keep],
            maximize      = objective.maximize,
            name          = objective.name,
            action_bounds = objective.action_bounds
        )
        gp = fit_gp(fold, noise, hypers)
        fold_mu, fold_std = gp.posterior_at(objective.xdata[i], raw=True)

        mu[i], std[i] = fold_mu[0, 0], fold_std[0, 0]
        models.append(gp)

    return mu, std, models


def _subset(X, n, kind, rng):
    """Indices of `n` rows of the (N, d) normalized actions `X`.

    'random' draws them iid. 'maximin' grows a set greedily from a random start,
    each step taking the row farthest from everything already chosen.
    """
    if kind == 'random':
        return rng.choice(len(X), size=n, replace=False)
    if kind != 'maximin':
        raise ValueError(f"kind must be 'random' or 'maximin', got {kind!r}")

    distance = squareform(pdist(X))
    keep = [int(rng.integers(len(X)))]
    while len(keep) < n:
        # a chosen row is at distance 0 from itself, so it is never re-picked
        keep.append(int(np.argmax(distance[keep].min(axis=0))))

    return np.array(keep)


def loo_curve(objective: Objective, fit_gp, noise, score, sizes=None,
              repeats=10, kind='random', seed=0, hypers=None):
    """Held-out score against dataset size, by subsampling the measurements.

    Each subset of size n is cross-validated with `loo`, so every fold behind
    its score was fit on n - 1 points. A curve still climbing at the full
    dataset means more measurements would help; one flat and low means the
    objective is noise-limited, and more of the same will not.

    Every subset is pinned to the full objective's `action_box()`, so the
    normalized frame -- and anything stated in it, a lengthscale floor included
    -- means the same thing at every size. Without that a smaller subset spans a
    smaller box, and the budget and the frame would vary together.

    Args:
        objective: the full set of measurements.
        fit_gp: builds one fold's fitted GP, called as `loo` calls it.
        noise: passed through to `fit_gp`.
        score: called as `score(y_true, y_pred)` on each subset's held-out
            predictions; sklearn's `r2_score` is the usual one. It is an
            argument so that scoring stays the caller's choice and no metric
            library enters the package.
        sizes: subset sizes to score. Defaults to `MIN_SIZE` through N. Below
            that a fold has fewer training points than the kernel has
            hyperparameters, and the score is the prior rather than a fit.
        repeats: subsets drawn per size. A size with fewer distinct subsets than
            that draws duplicates, which is honest -- the spread at the full
            dataset really is zero.
        kind: 'random' for iid subsets, 'maximin' for space-filling ones. A
            random subset of a space-filling design is clumped, so 'random' is
            the average case and 'maximin' is nearer a study run at that size.
        seed: seeds the subset draws.
        hypers: a `GPHyperparameters` to freeze in every fold, or None to refit.

    Returns:
        (sizes, scores): sizes (S,), scores (S, repeats). Subset size n means
        every fold behind that score was fit on n - 1 points.
    """
    total = objective.ydata.size
    sizes = np.arange(MIN_SIZE, total + 1) if sizes is None else np.asarray(sizes)
    if sizes.size == 0 or sizes.min() < 2 or sizes.max() > total:
        raise ValueError(f'subset sizes must lie in [2, {total}], got {sizes.tolist()}')

    rng    = np.random.default_rng(seed)
    bounds = objective.action_box()
    scores = np.empty((sizes.size, repeats))

    for i, n in enumerate(sizes):
        for j in range(repeats):
            keep   = _subset(objective.normalized_x, int(n), kind, rng)
            subset = Objective.from_data(
                actions       = objective.xdata[keep],
                values        = objective.ydata[keep],
                maximize      = objective.maximize,
                name          = objective.name,
                action_bounds = bounds
            )
            mu, _, _ = loo(subset, fit_gp, noise, hypers)
            scores[i, j] = score(subset.ydata, mu)

    return sizes, scores
