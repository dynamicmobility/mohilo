import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import LinAlgError

from pypolar.optimization.gp.base import GPModel


class LaplaceGP(GPModel):
    """Laplace-approximation GP for arbitrary (non-Gaussian) likelihoods.

    Finds the MAP reward vector by minimizing ``likelihood(r) + 0.5 r^T K^-1 r``
    and approximates the posterior covariance by the inverse Hessian at the mode.

    Use this for likelihoods without a closed-form posterior. For Gaussian 
    regression feedback use ``ConjugateGP``, which is more efficient.
    """

    def __init__(self, *args, x0_init_method="random", **kwargs):
        """
        Args:
            x0_init_method: initialization method for the optimizer start point
                ('random')
        """
        super().__init__(*args, **kwargs)
        
        # GP terms
        self.x0_init_method = x0_init_method
        self.x0 = None
        self.likelihood = None
        self.cov = None
        self.cov_inv = None
        self._posterior_chol = None
        self._hessian_chol = None
        self._const_hessian = None
        self._grad0 = None
        self._cov_key = None
        self._is_quadratic = False
        
        
    def compute_covariance(self, action_space):
        """Compute (and cache) the prior covariance and its inverse.
        """
        self.actions = action_space
        key = self._cache_key()
        if self._cov_key == key:
            return
        self.cov = self.prior_cov()
        self.cov_inv = np.linalg.inv(self.cov)
        self._cov_key = key

    def set_data(self, action_space, likelihood, quadratic=None):
        """Attach the action space and likelihood, and build the objective.

        Args:
            action_space: (N, d) array of discretized actions.
            likelihood: function accepting a reward vector over which the GP is
                defined and returning a scalar loss. Must be JAX-compatible.
            quadratic: whether the objective is quadratic (constant Hessian),
            passing in None results in auto-detects.
        """
        self.compute_covariance(action_space)
        if self.x0_init_method == "random":
            self.x0 = 2 * self.rng.random(self.actions.shape[0]) - 1
        else:
            raise ValueError(
                f"Invalid x0 init method: {self.x0_init_method}"
            )

        self.likelihood = likelihood

        # JAX-compatible objective: likelihood + GP prior
        jax_cov_inv = jnp.array(self.cov_inv)

        def _objective(r):
            return likelihood(r) + 0.5 * r @ jax_cov_inv @ r

        # Jit-decorate
        self._jax_objective = jax.jit(_objective)
        self._jax_jacobian  = jax.jit(jax.grad(_objective))
        self._jax_hessian   = jax.jit(jax.hessian(_objective))

        # Wrap with jnp <-> np conversions
        self.objective = lambda r: float(self._jax_objective(jnp.asarray(r)))
        self.jacobian  = lambda r: np.asarray(self._jax_jacobian(jnp.asarray(r)))
        self.hessian   = lambda r: np.asarray(self._jax_hessian(jnp.asarray(r)))

        self.detect_quadratic_objective(quadratic)

    def detect_quadratic_objective(self, quadratic=None):
        """Detect whether the objective is quadratic (constant Hessian). Enables
        closed form solve rather than scipy minimize.

        Args:
            quadratic: force the detection result (True/False), or None to
                auto-detect by checking whether the Hessian is r-independent.
        """
        n = self.actions.shape[0]
        r0 = np.zeros(n)
        H = self.hessian(r0)
        if quadratic is None:
            r1 = 2 * self.rng.random(n) - 1
            quadratic = np.allclose(H, self.hessian(r1))

        self._is_quadratic = bool(quadratic)
        self._hessian_chol = None
        self._const_hessian = None
        self._grad0 = None
        if not self._is_quadratic:
            return

        # Cache the constant Hessian, its Cholesky factor (reused for both the
        # mean solve and the posterior covariance), and the linear term grad0.
        # Note the factor is of the Hessian, i.e. the *precision*: recovering the
        # covariance from it takes a solve, not `L @ L.T`.
        self._grad0 = self.jacobian(r0)   # grad(0) = grad0 since grad(r) = H r + grad0
        try:
            self._hessian_chol = cho_factor(H)
        except LinAlgError:
            H = H + 1e-8 * np.eye(n)
            self._hessian_chol = cho_factor(H)
        self._const_hessian = H

    def fit(self, set_x0=True, **kwargs):
        """Find the MAP reward vector.

        When the objective is quadratic the minimizer has a closed form and is
        obtained by a single linear solve ``mu = -H^{-1} g0`` reusing the cached
        Cholesky factor of ``H`` — exact and independent of dimension.
        Otherwise it falls back to scipy's iterative solver with exact JAX
        gradients and Hessians.

        Args:
            set_x0: carry the result over as the next fit's warm start.
            **kwargs: passed to scipy.optimize.minimize (e.g. method, options).
                Ignored on the closed-form path.

        Returns:
            The optimized mean vector.
        """
        if self._is_quadratic:
            # grad(r) = H r + g0 = 0  =>  r = -H^{-1} g0  (exact minimizer)
            self.mu = cho_solve(self._hessian_chol, -self._grad0)
            if set_x0:
                self.x0 = self.mu
            return self.mu

        res = minimize(
            self.objective,
            self.x0,
            jac=self.jacobian,
            hess=self.hessian,
            **kwargs,
        )
        if set_x0:
            self.x0 = res.x

        self.mu = res.x
        return self.mu

    def posterior_cov(self, r=None):
        """Laplace-approximation posterior covariance over the reward vector.

        The fit objective is the negative log-posterior (negative log-likelihood
        plus the GP prior term ``0.5 r^T cov_inv r``). Its Hessian at the mode is
        the posterior precision matrix, so the posterior covariance is the
        inverse of that Hessian. Requires ``set_data()`` to have been called.

        Args:
            r: reward vector to linearize around (defaults to the current ``mu``)

        Returns:
            (N, N) posterior covariance matrix.
        """
        # Quadratic objective: the Hessian is constant, so the Laplace posterior
        # is exact and independent of r. Reuse the cached Cholesky factor of H
        # rather than re-evaluating and inverting the Hessian.
        if self._is_quadratic and self._hessian_chol is not None:
            n = self._const_hessian.shape[0]
            return cho_solve(self._hessian_chol, np.eye(n))
        if r is None:
            r = self.mu
        precision = self.hessian(r)
        return np.linalg.inv(precision)

    def std(self, r=None):
        """Per-action posterior standard deviation (Laplace approximation).

        Args:
            r: reward vector to linearize around (defaults to the current ``mu``)

        Returns:
            Length-N array of standard deviations, one per action.
        """
        var = np.diag(self.posterior_cov(r))
        return np.sqrt(np.clip(var, 0, None))

    def prepare_sampling(self):
        """Factorize the current Laplace posterior covariance.

        Unlike the prior Cholesky this depends on the data through the Hessian
        at the mode, so it is redone on each call.
        """
        self._posterior_chol = self.cholesky_factorization(self.posterior_cov())

    def sample_posterior(self, rng):
        """Draw one joint sample ``mu + L z`` from the posterior.

        ``L`` is the Cholesky factor of the posterior covariance.

        Args:
            rng: a numpy random Generator.

        Returns:
            Length-N sample of the reward vector.
        """
        if self._posterior_chol is None:
            self.prepare_sampling()
        return self.mu + self._posterior_chol @ rng.standard_normal(len(self.mu))
