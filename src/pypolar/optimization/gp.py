"""Gaussian process modelling of decoupled objectives, on BoTorch."""

import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

from pypolar.optimization.objectives import DecoupledObjectives, Objective

DTYPE           = torch.float64
NOISE_STD       = 0.05      # observation noise, as a fraction of each objective's spread
RAW_SAMPLES     = 512       # stage-1 Sobol samples per optimization
NUM_RESTARTS    = 8         # stage-2 L-BFGS-B starting points


def build_botorch_gp(
    actions       : np.ndarray,
    values        : np.ndarray,
    noise_std     : float,
    signal_var    : float,
    length_scale  : float
) -> SingleTaskGP:
    train_X = torch.as_tensor(actions, dtype=DTYPE)
    train_Y = torch.as_tensor(values, dtype=DTYPE).reshape(-1, 1)

    prior_covar = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1])).to(DTYPE)
    prior_covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
    prior_covar.outputscale = torch.tensor(signal_var, dtype=DTYPE)

    # Standardize divides train_Yvar by the same variance it divides train_Y
    # by, so pre-scaling by the spread leaves noise_std as a fraction of the
    # objective's own standard deviation rather than a raw magnitude.
    spread = train_Y.std() if train_Y.shape[0] > 1 else torch.ones((), dtype=DTYPE)

    return SingleTaskGP(
        train_X             = train_X,
        train_Y             = train_Y,
        train_Yvar          = torch.full_like(train_Y, (noise_std * spread) ** 2),
        covar_module        = prior_covar,
        mean_module         = ZeroMean(),
        outcome_transform   = Standardize(m=1),
        input_transform     = None,
    )

class BoTorchGP:
    """BoTorch GP in a convenient wrapper.
    """

    LENGTH_SCALE = 0.2    # 20% of the unit action box
    SIGNAL_VAR   = 1.0    # prior variance of unit-variance values
    
    def __init__(self, objective: Objective, noise_std, fit_hyperparameters=True):
        """
        Args:
            objective: the measurements to condition on.
            fit_hyperparameters: fit the lengthscales and signal variances by
                marginal likelihood, starting from LENGTH_SCALE and SIGNAL_VAR.
            noise_std: observation noise, as a fraction of each objective's
                own standard deviation.
        """
        self.fit_hyperparameters = fit_hyperparameters
        self.noise_std = noise_std
        self.update_feedback(objective)
        
    def update_feedback(self, objective: Objective):
        """Rebuild GP against the current feedback.

        Args:
            objective: the measurements to condition on.

        Returns:
            The `SingleTaskGP`, in eval mode.
        """
        self.objective = objective
        self.model = build_botorch_gp(
            actions      = objective.normalized_x,
            values       = objective.standard_y,
            noise_std    = self.noise_std,
            signal_var   = self.SIGNAL_VAR,
            length_scale = self.LENGTH_SCALE
        ).eval()

        if self.fit_hyperparameters:
            # one exact GP, so one exact marginal likelihood
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            )
            self.model.eval()

        return self.model

    def posterior_at(self, action, normalized=False, raw=False, chunk=2048):
        """Posterior mean and standard deviation at arbitrary actions.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.
            chunk: actions evaluated per posterior call.

        Returns:
            (mean, std), each (n, m), in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        X = np.atleast_2d(np.asarray(action, dtype=float))
        if not normalized:
            X = self.objective.xtransform(X)

        # fed as q=1 batch elements, so each posterior is 1x1 per objective and
        # gpytorch never forms the n x n test-test block
        X = torch.as_tensor(X, dtype=DTYPE).unsqueeze(1)
        with torch.no_grad():
            posteriors = [self.model.posterior(X[i:i + chunk])
                          for i in range(0, X.shape[0], chunk)]
            mu  = torch.cat([p.mean.squeeze(1)     for p in posteriors])
            var = torch.cat([p.variance.squeeze(1) for p in posteriors])

        mu, std = mu.numpy(), var.sqrt().numpy()
        return self.objective.to_raw(mu, std) if raw else (mu, std)

    def best_actions(self, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                     raw=False):
        """The action maximizing each objective's posterior mean.

        Args:
            num_restarts: L-BFGS-B starting points per objective.
            raw_samples: Sobol samples scanned to pick those starting points.
            raw: report the values in each objective's own units instead of
                maximization space. The actions are in raw units either way.

        Returns:
            (actions, mu, std): (m, d) actions in raw units, and the length-m
            posterior mean and standard deviation of each objective at its own
            best action, in maximization space unless `raw` is set.
        """
        d = self.objective.normalized_x.shape[1]
        bounds = torch.stack([torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)])

        # one single-output problem per objective, over the normalized box
        actions = np.vstack(
            optimize_acqf(
                acq_function    = PosteriorMean(self.model),
                bounds          = bounds,
                q               = 1,
                num_restarts    = num_restarts,
                raw_samples     = raw_samples
            )[0].detach().numpy()
        )

        # (m, m) evaluated at m actions; the diagonal is each objective at its own
        mu, std = self.posterior_at(actions, normalized=True, raw=raw)
        return self.objective.xtransform.inv(actions), np.diag(mu), np.diag(std)


class DecoupledMOGP:
    """Independent (decoupled) per-objective GPs over one shared action frame.
    """

    LENGTH_SCALE = 0.2    # 20% of the unit action box
    SIGNAL_VAR   = 1.0    # prior variance of unit-variance values

    def __init__(self, objectives: DecoupledObjectives, fit_hyperparameters=True,
                 noise_std=NOISE_STD):
        """
        Args:
            objectives: the measurements to condition on.
            fit_hyperparameters: fit the lengthscales and signal variances by
                marginal likelihood, starting from LENGTH_SCALE and SIGNAL_VAR.
            noise_std: observation noise, as a fraction of each objective's
                own standard deviation.
        """
        self.fit_hyperparameters = fit_hyperparameters
        self.noise_std = noise_std
        self.update_feedback(objectives)

    def update_feedback(self, objectives: DecoupledObjectives):
        """Rebuild every GP against the current feedback.

        Args:
            objectives: the measurements to condition on.

        Returns:
            The `ModelListGP`, in eval mode.
        """
        self.objectives = objectives
        self.model = ModelListGP(*[
            build_botorch_gp(
                actions      = objectives.actions(i),
                values       = objectives.feedback(i),
                noise_std    = self.noise_std,
                signal_var   = self.SIGNAL_VAR,
                length_scale = self.LENGTH_SCALE
            )
            for i in range(len(objectives))
        ]).eval()

        if self.fit_hyperparameters:
            # the sum splits over the sub-models because the objectives are
            # independent, so one call fits all of them
            fit_gpytorch_mll(
                SumMarginalLogLikelihood(self.model.likelihood, self.model)
            )
            self.model.eval()

        return self.model

    def posterior_at(self, action, normalized=False, raw=False, chunk=2048):
        """Posterior mean and standard deviation at arbitrary actions.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.
            chunk: actions evaluated per posterior call.

        Returns:
            (mean, std), each (n, m), in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        X = np.atleast_2d(np.asarray(action, dtype=float))
        if not normalized:
            X = self.objectives.xtransform(X)

        # fed as q=1 batch elements, so each posterior is 1x1 per objective and
        # gpytorch never forms the n x n test-test block
        X = torch.as_tensor(X, dtype=DTYPE).unsqueeze(1)
        with torch.no_grad():
            posteriors = [self.model.posterior(X[i:i + chunk])
                          for i in range(0, X.shape[0], chunk)]
            mu  = torch.cat([p.mean.squeeze(1)     for p in posteriors])
            var = torch.cat([p.variance.squeeze(1) for p in posteriors])

        mu, std = mu.numpy(), var.sqrt().numpy()
        return self.objectives.to_raw(mu, std) if raw else (mu, std)

    def best_actions(self, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                     raw=False):
        """The action maximizing each objective's posterior mean.

        Args:
            num_restarts: L-BFGS-B starting points per objective.
            raw_samples: Sobol samples scanned to pick those starting points.
            raw: report the values in each objective's own units instead of
                maximization space. The actions are in raw units either way.

        Returns:
            (actions, mu, std): (m, d) actions in raw units, and the length-m
            posterior mean and standard deviation of each objective at its own
            best action, in maximization space unless `raw` is set.
        """
        d = self.objectives.actions(0).shape[1]
        bounds = torch.stack([torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)])

        # one single-output problem per objective, over the normalized box
        actions = np.vstack([
            optimize_acqf(PosteriorMean(gp), bounds=bounds, q=1,
                          num_restarts=num_restarts, raw_samples=raw_samples
                          )[0].detach().numpy()
            for gp in self.model.models
        ])

        # (m, m) evaluated at m actions; the diagonal is each objective at its own
        mu, std = self.posterior_at(actions, normalized=True, raw=raw)
        return self.objectives.xtransform.inv(actions), np.diag(mu), np.diag(std)
