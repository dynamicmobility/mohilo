from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import LinAlgError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from pypolar.optimization.gp.kernels import make_kernel


class GPModel(ABC):
    """Base class for Gaussian process models over a discretized action space.

    Holds everything that does not depend on how the posterior is represented:
    the kernel algebra, the cached prior Cholesky factor used for sampling, and
    the polynomial fit of the mean.

    Subclasses differ only in how they turn feedback into a posterior, and each
    implements the same contract: ``set_data`` (backend-specific payload),
    then ``fit``, ``std``, ``posterior_cov``, ``prepare_sampling`` and
    ``sample_posterior``.
    """

    # Added to the prior covariance diagonal for numerical stability.
    JITTER = 1e-5

    def __init__(
        self,
        kernel="squared_exp",
        signal_variance=1,
        length_scale=1,
        rng=np.random.default_rng(),
    ):
        """
        Args:
            kernel: kernel name (e.g. 'squared_exp') or a kernel instance
            signal_variance: signal variance (dictates expected variation in the
                latent reward)
            length_scale: length scale (dictates the smoothness of the latent
                reward)
            rng: numpy random Generator
        """
        self.kernel = make_kernel(kernel, signal_variance, length_scale)
        self.rng = rng
        self.actions = None
        self.mu = None
        self._prior_chol = None
        self._prior_chol_key = None

    # ------------------------------------------------------------------
    # Kernel algebra
    # ------------------------------------------------------------------

    def kernel_cross(self, A, B):
        """Cross-covariance k(A, B) for an (n, d) and (m, d) set of points."""
        return self.kernel(A, B)

    def prior_var(self):
        """Prior variance k(x, x) + jitter, identical for every action."""
        return self.kernel.diag_var() + self.JITTER

    def prior_cov(self):
        """Jittered prior covariance over the full action space, (N, N)."""
        cov = self.kernel(self.actions, self.actions)
        np.fill_diagonal(cov, np.diagonal(cov) + self.JITTER)
        return cov

    def _cache_key(self):
        """Identity of the state the prior covariance depends on. Used as a 
        check in many places to see if any core variables need updating."""
        return (id(self.actions), self.actions.shape, self.kernel.key())

    @staticmethod
    def cholesky_factorization(A, max_tries=5):
        """Cholesky factorization of (positive semi-definite) matrix A. 
        
        Attempts the factorization directly and only adds jitter if it fails.

        Args:
            A: (N, N) symmetric matrix.
            max_tries: number of jitter escalations before giving up.

        Returns:
            Lower-triangular L with L @ L.T ~= A.
        """
        # try the decomposition on its own
        try:
            return np.linalg.cholesky(A)
        except LinAlgError:
            pass

        # if the above failed, progressively add jitter along the diagonal until
        # out of tries or a successful factorization is obtained
        n = A.shape[0]
        jitter = 1e-10 * max(np.abs(np.diagonal(A)).max(), 1.0)
        for _ in range(max_tries):
            try:
                return np.linalg.cholesky(A + jitter * np.eye(n))
            except LinAlgError:
                jitter *= 10
        raise LinAlgError("matrix is not positive definite even with jitter")

    def _ensure_prior_chol(self):
        """Compute (and cache) the Cholesky factor of the prior covariance.
        """
        key = self._cache_key()
        if self._prior_chol_key != key:
            self._prior_chol = self.cholesky_factorization(self.prior_cov())
            self._prior_chol_key = key
        return self._prior_chol

    # ------------------------------------------------------------------
    # Backend contract
    # ------------------------------------------------------------------

    @abstractmethod
    def set_data(self, action_space, *args, **kwargs):
        """Attach an action space and the feedback observed so far.

        The payload is backend-specific — see each subclass — but every backend
        is usable through the shared methods below once this has been called.
        """

    @abstractmethod
    def fit(self, set_x0=True, **kwargs):
        """Compute the posterior mean. Sets and returns ``self.mu``."""

    @abstractmethod
    def posterior_cov(self, r=None):
        """Full (N, N) posterior covariance."""

    @abstractmethod
    def std(self, r=None):
        """Length-N array of per-action posterior standard deviations."""

    def posterior_cov_cross(self, idx):
        """Posterior covariance between every action and ``actions[idx]``, (N, C).

        Args:
            idx: length-C array of action indices.

        Returns:
            (N, C) block of the posterior covariance.
        """
        return self.posterior_cov()[:, np.asarray(idx, dtype=int)]

    @abstractmethod
    def prepare_sampling(self):
        """Prepare the state needed by ``sample_posterior()``. Call after fit."""

    @abstractmethod
    def sample_posterior(self, rng):
        """Draw one joint sample of the reward vector from the posterior."""

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------

    def functionalize(self, degree=2):
        """Fits a polynomial to the GP mean for continuous evaluation.

        Args:
            degree: polynomial degree

        Returns:
            Polynomial predictions at the action space points.
        """
        self.poly = PolynomialFeatures(degree=degree)
        X_poly = self.poly.fit_transform(self.actions)
        model = LinearRegression()
        model.fit(X_poly, self.mu)

        y_pred = model.predict(self.poly.transform(self.actions))
        self.f = lambda x: model.predict(self.poly.transform(x))
        return y_pred.reshape(len(self.actions))
