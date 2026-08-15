"""Gaussian process modelling of decoupled objectives, on BoTorch."""

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from pypolar.optimization.objectives import DecoupledObjectives, Objective

DTYPE           = torch.float64
NOISE_STD       = 0.05      # observation noise, as a fraction of each objective's spread
LENGTH_SCALE    = 0.2       # 20% of the unit action box
SIGNAL_VAR      = 1.0       # prior variance of unit-variance values
PRIOR_SIGMA     = 1.0       # LogNormal noise prior width, matching BoTorch's own
RAW_SAMPLES     = 512       # stage-1 Sobol samples per optimization
NUM_RESTARTS    = 8         # stage-2 L-BFGS-B starting points


@dataclass(frozen=True)
class NoiseModel:
    """How a GP treats its observation noise, as a fraction of the objective's
    own standard deviation.

    Three representations, one per constructor:

    - **`pinned(std)`** fixes the noise through `train_Yvar`, which selects
      `FixedNoiseGaussianLikelihood`, so a fit only ever moves the kernel.
    - **`fitted()`** leaves it free for the marginal likelihood. On a small
      design this can collapse onto interpolation: the GP threads every
      measurement, calls the residual zero, and puts its argmax on whichever
      point drew the luckiest noise.
    - **`prior(median, sigma)`** fits it under a LogNormal centered on `median`,
      pulling a collapsing fit back without forbidding any value outright. This
      is the conventional choice, and what BoTorch's own default likelihood does.

    A pinned noise is never fitted, so it cannot also carry a prior; the two
    constructors are mutually exclusive and the combination raises.

    Attributes:
        std: the pinned level, or None when the noise is fitted.
        median: center of the prior on the noise *standard deviation*, or None
            to fit it unpriored.
        sigma: prior width in log space, on the variance.
    """

    std    : float | None = None
    median : float | None = None
    sigma  : float        = PRIOR_SIGMA

    def __post_init__(self):
        if self.std is not None and self.median is not None:
            raise ValueError('a pinned noise is never fitted, so it cannot carry '
                             'a prior; use NoiseModel.pinned or .prior, not both')
        if self.std is not None and self.std < 0:
            raise ValueError('noise std must be non-negative')
        if self.median is not None and self.median <= 0:
            raise ValueError('a LogNormal prior has positive support, so its '
                             'median must be positive')

    @classmethod
    def pinned(cls, std: float) -> 'NoiseModel':
        return cls(std=std)

    @classmethod
    def fitted(cls) -> 'NoiseModel':
        return cls()

    @classmethod
    def prior(cls, median: float, sigma: float = PRIOR_SIGMA) -> 'NoiseModel':
        return cls(median=median, sigma=sigma)

    @classmethod
    def coerce(cls, noise) -> 'NoiseModel':
        """A `NoiseModel` unchanged, a number as `pinned`, None as `fitted`."""
        if isinstance(noise, cls):
            return noise
        return cls.fitted() if noise is None else cls.pinned(noise)

    @property
    def is_fitted(self) -> bool:
        return self.std is None

    def train_yvar(self, train_Y: torch.Tensor, spread: torch.Tensor):
        """The per-point variance to pin, or None when the noise is fitted.

        `Standardize` divides `train_Yvar` by the same variance it divides
        `train_Y` by, so pre-scaling by the spread is what leaves `std` a
        fraction of the objective's own standard deviation rather than a raw
        magnitude.
        """
        if self.std is None:
            return None
        return torch.full_like(train_Y, (self.std * spread) ** 2)

    def likelihood(self):
        """The likelihood to attach, or None to let `train_Yvar` pick a fixed one.

        Passed explicitly whenever the noise is fitted, because BoTorch's default
        likelihood carries a LogNormal noise prior of its own, centered low
        enough to decide the answer on a small design.
        """
        if self.std is not None:
            return None

        # post-Standardize the noise variance is the squared fraction, so a
        # median noise standard deviation of m is a median variance of m^2
        prior = None if self.median is None else LogNormalPrior(
            loc=2.0 * float(np.log(self.median)), scale=self.sigma
        )
        return GaussianLikelihood(noise_prior=prior).to(DTYPE)


def build_botorch_gp(
    actions          : np.ndarray,
    values           : np.ndarray,
    noise            : 'NoiseModel | float | None',
    signal_var       : float,
    length_scale     : float,
    min_length_scale : float | None = None
) -> SingleTaskGP:
    """The one `SingleTaskGP` configuration the package uses.

    Args:
        actions: (n, d) actions, in the normalized frame.
        values: (n,) values, already standardized and larger-is-better.
        noise: a `NoiseModel`, or the shorthands it coerces -- a number for a
            pinned fraction of the values' own spread, None to fit the noise.
        signal_var: starting ScaleKernel outputscale.
        length_scale: starting ARD lengthscale, scalar or one per dimension.
        min_length_scale: lower bound on every lengthscale, in the normalized
            frame. None leaves them bounded only away from zero.

    Returns:
        the `SingleTaskGP`, unfitted.
    """
    train_X = torch.as_tensor(actions, dtype=DTYPE)
    train_Y = torch.as_tensor(values, dtype=DTYPE).reshape(-1, 1)
    noise   = NoiseModel.coerce(noise)

    prior_covar = ScaleKernel(RBFKernel(
        ard_num_dims           = train_X.shape[-1],
        lengthscale_constraint = None if min_length_scale is None
                                 else GreaterThan(min_length_scale)
    )).to(DTYPE)
    if min_length_scale is not None:
        # a GreaterThan cannot represent its own edge, so start strictly inside.
        # elementwise, since length_scale may be one per action dimension
        length_scale = np.maximum(length_scale, 1.5 * min_length_scale)
    prior_covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
    prior_covar.outputscale = torch.tensor(signal_var, dtype=DTYPE)

    # one point has no sample standard deviation, so fall back to 1 rather than
    # propagate a NaN into the likelihood
    spread = train_Y.std() if train_Y.shape[0] > 1 else torch.ones((), dtype=DTYPE)

    return SingleTaskGP(
        train_X             = train_X,
        train_Y             = train_Y,
        train_Yvar          = noise.train_yvar(train_Y, spread),
        likelihood          = noise.likelihood(),
        covar_module        = prior_covar,
        mean_module         = ZeroMean(),
        outcome_transform   = Standardize(m=1),
        input_transform     = None,
    )

@dataclass(frozen=True, eq=False)
class GPHyperparameters:
    """Every hyperparameter of one GP built by `build_botorch_gp`.

    Attributes:
        lengthscale: (d,) ARD lengthscales, in the normalized action frame.
            Scalar when the same value is meant for every dimension.
        signal_var: the ScaleKernel outputscale.
        noise_var: the observation noise, pinned or fitted.
        standardize_scale: the divisor Standardize applied to the values. Both
            variances are in post-Standardize units; multiplying them by
            `standardize_scale ** 2` puts them in the units the GP was handed.
            Defaults to 1, which is the scale of values already at unit spread.
    """

    lengthscale       : np.ndarray | float
    signal_var        : float
    noise_var         : float
    standardize_scale : float = 1.0


def gp_hyperparameters(model: SingleTaskGP) -> GPHyperparameters:
    """The hyperparameters of one GP built by `build_botorch_gp`.

    Args:
        model: a GP built by `build_botorch_gp`.

    Returns:
        the `GPHyperparameters`, read off the model's own tensors.
    """
    kernel = model.covar_module
    return GPHyperparameters(
        lengthscale       = kernel.base_kernel.lengthscale.detach().numpy().ravel(),
        signal_var        = kernel.outputscale.item(),
        # identical at every point by construction, so the mean is that value
        noise_var         = model.likelihood.noise.mean().item(),
        standardize_scale = model.outcome_transform.stdvs.item(),
    )


class BoTorchGP:
    """BoTorch GP in a convenient wrapper.
    """

    def __init__(self, objective: Objective, noise, fit_hyperparameters=True,
                 length_scale=LENGTH_SCALE, signal_var=SIGNAL_VAR,
                 min_length_scale=None):
        """
        Args:
            objective: the measurements to condition on.
            noise: a `NoiseModel`, or the shorthands it coerces -- a number for
                a pinned fraction of the objective's own standard deviation,
                None to fit the noise.
            fit_hyperparameters: fit the lengthscales and signal variance by
                marginal likelihood, starting from `length_scale`/`signal_var`.
            length_scale: starting ARD lengthscale, in the normalized frame.
            signal_var: starting ScaleKernel outputscale.
            min_length_scale: lower bound on every lengthscale, or None.
        """
        self.noise = NoiseModel.coerce(noise)
        if self.noise.is_fitted and not fit_hyperparameters:
            raise ValueError('a fitted noise is determined by the marginal '
                             'likelihood, so fit_hyperparameters must be True')
        self.fit_hyperparameters = fit_hyperparameters
        self.length_scale     = length_scale
        self.signal_var       = signal_var
        self.min_length_scale = min_length_scale
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
            actions          = objective.normalized_x,
            values           = objective.standard_y,
            noise            = self.noise,
            signal_var       = self.signal_var,
            length_scale     = self.length_scale,
            min_length_scale = self.min_length_scale
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

    def sample_paths(self, action, num_paths, normalized=False, raw=False):
        """Sample paths of the posterior over arbitrary actions.

        Drawn from the *joint* posterior over the whole set of actions, which is
        what makes each draw a function: sampling every action from its own
        marginal independently would discard the covariance between them and
        return white noise. That is also why this cannot chunk the way
        `posterior_at` does -- the n x n test-test block it avoids forming is
        the very object a joint draw needs.

        The draws come from torch's global generator, so seeding that is what
        makes them repeatable.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            num_paths: paths drawn. Not botorch's `q`, which here is n: the
                actions are one q-batch, and this is how often it is sampled.
            normalized: True if `action` is already in the [0, 1]^d frame.
            raw: report in the objective's own units instead of maximization
                space. This undoes the sign too.

        Returns:
            (num_paths, n, m) draws, in maximization space: larger-is-better and
            standardized, matching `objective.standard_y`. With `raw=True`, in
            the units the measurements were taken in.
        """
        X = np.atleast_2d(np.asarray(action, dtype=float))
        if not normalized:
            X = self.objective.xtransform(X)

        # fed as one batch element rather than n of them, the opposite of
        # `posterior_at`, so the test-test covariance is formed and sampled
        X = torch.as_tensor(X, dtype=DTYPE)
        with torch.no_grad():
            paths = self.model.posterior(X).rsample(torch.Size([num_paths])).numpy()

        return self.objective.to_raw(paths) if raw else paths

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
    
    def get_fitted_hyperparameters(self):
        """The GP's hyperparameters, as described by `gp_hyperparameters`."""
        return gp_hyperparameters(self.model)


class DecoupledMOGP:
    """Independent (decoupled) per-objective GPs over one shared action frame.
    """

    def __init__(self, objectives: DecoupledObjectives, fit_hyperparameters=True,
                 noise=NOISE_STD, length_scale=LENGTH_SCALE,
                 signal_var=SIGNAL_VAR, min_length_scale=None):
        """
        Args:
            objectives: the measurements to condition on.
            fit_hyperparameters: fit the lengthscales and signal variances by
                marginal likelihood, starting from `length_scale`/`signal_var`.
            noise: a `NoiseModel`, or the shorthands it coerces -- a number for
                a pinned fraction of each objective's own standard deviation,
                None to fit the noise. Shared by every sub-model, though each
                fits its own value from its own term of the likelihood.
            length_scale: starting ARD lengthscale, in the normalized frame.
            signal_var: starting ScaleKernel outputscale.
            min_length_scale: lower bound on every lengthscale, or None.
        """
        self.noise = NoiseModel.coerce(noise)
        if self.noise.is_fitted and not fit_hyperparameters:
            raise ValueError('a fitted noise is determined by the marginal '
                             'likelihood, so fit_hyperparameters must be True')
        self.fit_hyperparameters = fit_hyperparameters
        self.length_scale     = length_scale
        self.signal_var       = signal_var
        self.min_length_scale = min_length_scale
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
                actions          = objectives.actions(i),
                values           = objectives.feedback(i),
                noise            = self.noise,
                signal_var       = self.signal_var,
                length_scale     = self.length_scale,
                min_length_scale = self.min_length_scale
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

    def sample_paths(self, action, num_paths, normalized=False, raw=False):
        """Sample paths of the posterior over arbitrary actions.

        Drawn from the *joint* posterior over the whole set of actions, which is
        what makes each draw a function: sampling every action from its own
        marginal independently would discard the covariance between them and
        return white noise. That is also why this cannot chunk the way
        `posterior_at` does -- the n x n test-test block it avoids forming is
        the very object a joint draw needs.

        The objectives are decoupled, so a draw is joint over the actions but
        independent across the objectives: column j of every path comes from
        objective j's own GP and carries no covariance with column k.

        The draws come from torch's global generator, so seeding that is what
        makes them repeatable.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            num_paths: paths drawn. Not botorch's `q`, which here is n: the
                actions are one q-batch, and this is how often it is sampled.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.

        Returns:
            (num_paths, n, m) draws, in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        X = np.atleast_2d(np.asarray(action, dtype=float))
        if not normalized:
            X = self.objectives.xtransform(X)

        # fed as one batch element rather than n of them, the opposite of
        # `posterior_at`, so the test-test covariance is formed and sampled
        X = torch.as_tensor(X, dtype=DTYPE)
        with torch.no_grad():
            paths = self.model.posterior(X).rsample(torch.Size([num_paths])).numpy()

        return self.objectives.to_raw(paths) if raw else paths

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

    def get_fitted_hyperparameters(self):
        """Each objective's GP hyperparameters, in objective order.

        Returns:
            a length-m list of `GPHyperparameters`. The objectives are
            decoupled, so no entry is shared between them.
        """
        return [gp_hyperparameters(gp) for gp in self.model.models]
