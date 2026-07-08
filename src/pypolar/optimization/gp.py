from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class BasicGP:
    """Gaussian process model for approximating a latent reward function.

    Uses JAX for automatic differentiation of the objective, providing
    exact gradients and Hessians to scipy.optimize.minimize.
    """

    def __init__(
        self,
        kernel="squared_exp",
        signal_variance=1,
        length_scale=1,
        x0_init_method="random",
        rng=np.random.default_rng()
    ):
        """Approximates an unknown latent reward function using a Gaussian
        process using a provided objective function.

        Args:
            kernel: kernel type (currently only 'squared_exp')
            signal_variance: signal variance (dictates expected variation in the latent reward)
            length_scale: length scale (dictates the smoothness of the latent reward)
            mu_init_method: initialization method for the mean ('random')
        """
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.x0_init_method = x0_init_method
        self.mu = None
        self.x0 = None
        self.f = None
        self.rng = rng
        if kernel == "squared_exp":
            self.kernel = self.squared_exp_kernel
        self._cov_key = None

    def _ensure_cov(self, action_space):
        """Compute (and cache) the prior covariance and its inverse.

        The prior covariance depends only on the action space and kernel
        hyperparameters, so it is recomputed only when those change — repeated
        setup() calls on the same action space reuse the cached inverse rather
        than redoing an O(N^3) matrix inversion each time.
        """
        key = (id(action_space), getattr(action_space, "shape", None),
               self.signal_variance, self.length_scale)
        if self._cov_key == key:
            return
        self.actions = action_space
        self.cov = self.prior_cov()
        np.fill_diagonal(self.cov, np.diagonal(self.cov) + 1e-5)
        self.cov_inv = np.linalg.inv(self.cov)
        self._cov_key = key

    def setup(self, action_space, likelihood):
        """Initialize the GP with an action space and likelihood function.

        Automatically derives the Jacobian and Hessian of the full objective
        (likelihood + GP prior) using JAX autodiff. All three are JIT-compiled
        and wrapped for scipy compatibility.

        Args:
            action_space: array of discretized actions
            likelihood: function to minimize. accepts a reward vector over which
            the GP is defined and returns a scalar loss. Should be compatible with
            JAX (e.g. use jax.numpy instead of numpy).
        """
        self.actions = action_space
        self._ensure_cov(action_space)
        if self.x0_init_method == "random":
            self.x0 = 2 * self.rng.random(self.actions.shape[0]) - 1
        else:
            raise ValueError(f"Invalid mu init method: {self.mu_init_method}")

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

    def squared_exp_kernel(self, X):
        """Computes the squared exponential kernel matrix.

        Args:
            X: input data array of shape (n, d)

        Returns:
            Kernel matrix of shape (n, n).
        """
        sqdist = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)
            - 2 * np.dot(X, X.T)
        )
        K = self.signal_variance**2 * np.exp(-0.5 * sqdist / self.length_scale**2)
        return K

    def prior_cov(self):
        """Computes the prior covariance matrix using the kernel."""
        return self.kernel(self.actions)

    def fit(self, set_x0=True, **kwargs):
        """Fits the Gaussian process to minimize the likelihood.

        Gradients and Hessians are always available via JAX autodiff.

        Args:
            **kwargs: passed to scipy.optimize.minimize (e.g. method, options)

        Returns:
            The optimized mean vector.
        """
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
        inverse of that Hessian. Requires ``fit()`` (or at least ``setup()``) to
        have been called.

        Args:
            r: reward vector to linearize around (defaults to the current ``mu``)

        Returns:
            (N, N) posterior covariance matrix.
        """
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

        X_grid = self.actions
        y_pred = model.predict(self.poly.transform(X_grid))
        self.f = lambda x: model.predict(self.poly.transform(x))
        return y_pred.reshape(len(self.actions))
