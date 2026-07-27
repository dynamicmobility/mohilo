import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise import draw_matheron_paths
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.linalg import cho_factor, cho_solve

from pypolar.optimization.gp.base import GPModel


class BoTorchGP(GPModel):
    """Exact GP posterior for Gaussian (regression) feedback, backed by BoTorch.

    Wraps ``botorch.models.SingleTaskGP`` over the discretized action space.
    The observation noise is pinned from the likelihood precision, so the kernel
    hyperparameters are the only free ones: ``fit_hypers`` fits them by marginal
    likelihood on every ``fit()``, and ``ard`` gives the length scale one value
    per action dimension.

    The posterior mean, standard deviation and cross-covariance are evaluated in
    *data* space from the model's own covariance module: the only system solved
    is ``G = K_XX + sigma^2 I``, of size M x M where M is the number of feedback
    points, which keeps them O(N M^2) in time and O(N M) in memory over the
    length-N action space.

    Everything runs on CPU in float64; numpy is the boundary in both directions.
    """

    DTYPE = torch.float64

    def __init__(self, *args, fit_hypers=False, ard=False, **kwargs):
        """
        Args:
            fit_hypers: fit the kernel hyperparameters by marginal likelihood on
                each ``fit()``, starting from ``signal_variance`` and
                ``length_scale``.
            ard: give the length scale one value per action dimension.
            *args, **kwargs: see ``GPModel``.
        """
        super().__init__(*args, **kwargs)
        self.fit_hypers = fit_hypers
        self.ard = ard
        self.idx = None
        self.y = None
        self.sigma2 = None
        self.model = None
        self._grid = None
        self._grid_key = None
        self._blocks = None

    def _grid_tensor(self):
        """The action space as a float64 tensor, cached across ``set_data`` calls."""
        key = (id(self.actions), self.actions.shape)
        if self._grid_key != key:
            self._grid = torch.as_tensor(np.asarray(self.actions), dtype=self.DTYPE)
            self._grid_key = key
        return self._grid

    def _kernel_block(self, A, B):
        """Cross-covariance k(A, B) under the model's current hyperparameters."""
        with torch.no_grad():
            return self.model.covar_module(A, B).to_dense().numpy()

    def _data_blocks(self):
        """``(K_sX, chol(K_XX + sigma^2 I))``, the shared data-space factors.

        Discarded whenever the feedback or the hyperparameters change.

        Returns:
            The (N, M) cross-covariance and the M x M Gram factor.
        """
        if self._blocks is None:
            grid = self._grid_tensor()
            train = grid[self.idx]
            K_XX = self._kernel_block(train, train)
            self._blocks = (
                self._kernel_block(grid, train),
                cho_factor(K_XX + self.sigma2 * np.eye(self.idx.shape[0])),
            )
        return self._blocks

    def _covar_module(self, d):
        """A scaled RBF kernel carrying the configured hyperparameters.

        Args:
            d: number of action dimensions.

        Returns:
            A ``ScaleKernel`` in float64, so that assigning the hyperparameters
            stores them at full precision.
        """
        base = RBFKernel(ard_num_dims=d if self.ard else None)
        covar = ScaleKernel(base).to(self.DTYPE)
        covar.base_kernel.lengthscale = self.kernel.length_scale
        covar.outputscale = self.kernel.signal_variance ** 2
        return covar

    def set_data(self, action_space, idx, y, precision):
        """Attach the action space and regression feedback.

        Args:
            action_space: (N, d) array of discretized actions.
            idx: length-M array of fed-back action indices.
            y: length-M array of observed values.
            precision: likelihood precision ``lambda``, where the likelihood is
                ``lambda * ||S r - y||^2``.
        """
        self.actions = action_space
        idx = np.asarray(idx, dtype=int)
        y = np.asarray(y, dtype=float)

        # L = lambda*||Sr-y||^2 is a Gaussian NLL with sigma^2 = 1/(2*lambda).
        self.sigma2 = 1.0 / (2.0 * float(np.asarray(precision).item()))
        self.idx = idx
        self.y = y
        self._blocks = None

        if idx.shape[0] == 0:
            # SingleTaskGP needs at least one training point; the prior stands in
            # until the first feedback arrives.
            self.model = None
            return

        grid = self._grid_tensor()
        train_Y = torch.as_tensor(y, dtype=self.DTYPE).reshape(-1, 1)
        self.model = SingleTaskGP(
            train_X=grid[idx],
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, self.sigma2),
            covar_module=self._covar_module(action_space.shape[1]),
            # TODO: a ConstantMean fitted alongside `fit_hypers` would let the
            # prior follow off-centre feedback, which is what
            # MaxValueEntropySampler needs in order to explore.
            mean_module=ZeroMean(),
            # The default is Standardize, which would rescale `mu` out from
            # under the hyperparameters set above.
            outcome_transform=None,
            input_transform=None,
        )
        self.model.eval()

    def fit(self, set_x0=True, **kwargs):
        """Compute the posterior mean ``mu = K_sX (K_XX + sigma^2 I)^-1 y``.

        Args:
            set_x0: accepted for interface compatibility; the closed form needs
                no warm start.
            **kwargs: accepted and ignored (e.g. scipy ``method``/``options``
                passed by callers that also drive the Laplace backend).

        Returns:
            The posterior mean vector.
        """
        if self.model is None:
            self.mu = np.zeros(self.actions.shape[0])
            return self.mu

        if self.fit_hypers:
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            )
            self.model.eval()
            self._blocks = None

        K_sX, gram_chol = self._data_blocks()
        self.mu = K_sX @ cho_solve(gram_chol, self.y)
        return self.mu

    def posterior_cov(self, r=None):
        """Full (N, N) posterior covariance, straight from the model.

        Costs O(N^2) time and memory — prefer ``std()`` when only the diagonal is
        needed, and ``posterior_cov_cross()`` for a few columns.

        Args:
            r: accepted for interface compatibility; the posterior is exact and
                does not depend on a linearization point.

        Returns:
            (N, N) posterior covariance matrix.
        """
        if self.model is None:
            return self.prior_cov()
        with torch.no_grad():
            cov = self.model.posterior(self._grid_tensor()).distribution
            return cov.covariance_matrix.numpy()

    def posterior_cov_cross(self, idx):
        """Posterior covariance between every action and ``actions[idx]``, (N, C).

        Evaluated in data space as ``K_sc - K_sX G^-1 K_Xc`` with
        ``G = K_XX + sigma^2 I``, in O(N M C), without forming the N x N
        posterior covariance. The kernel blocks come from the model's own
        covariance module, so they follow any hyperparameters ``fit()`` learned.

        Args:
            idx: length-C array of action indices.

        Returns:
            (N, C) block of the posterior covariance.
        """
        idx = np.asarray(idx, dtype=int)
        if self.model is None:
            K_sc = self.kernel_cross(self.actions, self.actions[idx])
            K_sc[idx, np.arange(idx.shape[0])] += self.JITTER
            return K_sc

        grid = self._grid_tensor()
        K_sc = self._kernel_block(grid, grid[idx])
        K_Xc = self._kernel_block(grid[self.idx], grid[idx])
        K_sX, gram_chol = self._data_blocks()
        return K_sc - K_sX @ cho_solve(gram_chol, K_Xc)

    def std(self, r=None):
        """Per-action posterior standard deviation of the latent reward.

        Computes the variance diagonal directly, in O(N M^2), rather than
        building the full N x N posterior covariance to take its diagonal.

        Args:
            r: accepted for interface compatibility; unused.

        Returns:
            Length-N array of standard deviations, one per action.
        """
        if self.model is None:
            return np.full(self.actions.shape[0], np.sqrt(self.prior_var()))

        K_sX, gram_chol = self._data_blocks()
        prior_var = self.model.covar_module.outputscale.detach().item()
        var = prior_var - np.einsum(
            "ij,ji->i", K_sX, cho_solve(gram_chol, K_sX.T)
        )
        return np.sqrt(np.clip(var, 0, None))

    def prepare_sampling(self):
        """No state to prepare: each draw builds its own Matheron path.

        A path is fixed once drawn, and callers draw several times per fit
        (``MaxValueEntropySampler`` once per sample of ``f*``), so the draw
        belongs in ``sample_posterior()``.
        """

    def sample_posterior(self, rng):
        """Draw one joint sample of the reward vector.

        Uses Matheron's rule with random Fourier features for the prior term, so
        no N x N factorization is involved and the cost does not grow when
        ``fit_hypers`` changes the kernel every iteration.

        Args:
            rng: a numpy random Generator.

        Returns:
            Length-N sample of the reward vector.
        """
        if self.model is None:
            L = self._ensure_prior_chol()
            return L @ rng.standard_normal(self.actions.shape[0])

        # BoTorch draws from torch's global generator; forking keeps the draw
        # reproducible from `rng` alone and leaves that generator untouched.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(int(rng.integers(2 ** 32)))
            path = draw_matheron_paths(self.model, torch.Size([1]))
            sample = path(self._grid_tensor())
        return sample.reshape(-1).numpy()
