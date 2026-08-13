"""Leave-one-out cross-validation of a GP against its own measurements."""

import numpy as np

from pypolar.optimization.objectives import Objective


def loo(objective: Objective, fit_gp, noise, hypers=None):
    """Held-out posterior at every measured action.

    Fold i is fit on every measurement except i, and is then asked to predict
    point i. No fold ever sees the point it is scored against, so the residuals
    are out-of-sample errors rather than the training fit.

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
            actions  = objective.xdata[keep],
            values   = objective.ydata[keep],
            maximize = objective.maximize,
            name     = objective.name
        )
        gp = fit_gp(fold, noise, hypers)
        fold_mu, fold_std = gp.posterior_at(objective.xdata[i], raw=True)

        mu[i], std[i] = fold_mu[0, 0], fold_std[0, 0]
        models.append(gp)

    return mu, std, models
