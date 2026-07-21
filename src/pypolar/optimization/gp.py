from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import LinAlgError
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
        # Closed-form fast path state (populated by setup() -> _detect_quadratic)
        self._is_quadratic = False
        self._quad_chol = None
        self._quad_H = None
        self._quad_g0 = None

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

    def setup(self, action_space, likelihood, quadratic=None):
        """Initialize the GP with an action space and likelihood function.

        Automatically derives the Jacobian and Hessian of the full objective
        (likelihood + GP prior) using JAX autodiff. All three are JIT-compiled
        and wrapped for scipy compatibility.

        Args:
            action_space: array of discretized actions
            likelihood: function to minimize. accepts a reward vector over which
            the GP is defined and returns a scalar loss. Should be compatible with
            JAX (e.g. use jax.numpy instead of numpy).
            quadratic: whether the objective is quadratic (constant Hessian),
            enabling the closed-form ``fit()``. ``None`` (default) auto-detects;
            pass ``True``/``False`` to force it (e.g. skip the detection cost).
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

        self._detect_quadratic(quadratic)

    def _detect_quadratic(self, quadratic=None):
        """Detect whether the objective is quadratic (constant Hessian).

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
        self._quad_chol = None
        self._quad_H = None
        self._quad_g0 = None
        if not self._is_quadratic:
            return

        # Cache the constant Hessian, its Cholesky factor (reused for both the
        # mean solve and the posterior covariance), and the linear term g0.
        self._quad_g0 = self.jacobian(r0)   # grad(0) = g0 since grad(r) = H r + g0
        try:
            self._quad_chol = cho_factor(H)
        except LinAlgError:
            H = H + 1e-8 * np.eye(n)
            self._quad_chol = cho_factor(H)
        self._quad_H = H

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

        When the objective is quadratic (regression / HILO likelihoods), the
        minimizer has a closed form and is obtained by a single linear solve
        ``mu = -H^{-1} g0`` reusing the cached Cholesky factor of ``H`` — exact
        and independent of dimension. Otherwise (e.g. the sigmoid PBL
        likelihood) it falls back to scipy's iterative solver with exact JAX
        gradients and Hessians.

        Args:
            **kwargs: passed to scipy.optimize.minimize (e.g. method, options).
                Ignored on the closed-form path.

        Returns:
            The optimized mean vector.
        """
        if self._is_quadratic:
            # grad(r) = H r + g0 = 0  =>  r = -H^{-1} g0  (exact minimizer)
            self.mu = cho_solve(self._quad_chol, -self._quad_g0)
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
        inverse of that Hessian. Requires ``fit()`` (or at least ``setup()``) to
        have been called.

        Args:
            r: reward vector to linearize around (defaults to the current ``mu``)

        Returns:
            (N, N) posterior covariance matrix.
        """
        # Quadratic objective: the Hessian is constant, so the Laplace posterior
        # is exact and independent of r. Reuse the cached Cholesky factor of H
        # rather than re-evaluating and inverting the Hessian.
        if self._is_quadratic and self._quad_chol is not None:
            n = self._quad_H.shape[0]
            return cho_solve(self._quad_chol, np.eye(n))
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
    

class MultiObjectiveGP:
    
    def __init__(
        self,
        num_objs=1,
        kernels=['squared_exp'],
        signal_variances=[1],
        length_scales=[1],
        x0_init_methods=['random'],
        rng=np.random.default_rng()
    ):
        self.gps: list[BasicGP] = []
        for i in range(num_objs):
            gp = BasicGP(
                kernel=kernels[i],
                signal_variance=signal_variances[i],
                length_scale=length_scales[i],
                x0_init_method=x0_init_methods[i],
                rng=rng
            )
            self.gps.append(gp)
            
    def setup(self, action_space, likelihoods):
        for i in range(len(self.gps)):
            self.gps[i].setup(action_space, likelihoods[i])
            
    def fit(self, set_x0=True, **kwargs):
        for gp in self.gps:
            gp.fit(set_x0=set_x0, **kwargs)
        
        return [gp.mu for gp in self.gps]
    
    @property
    def std(self, r=None):
        return [gp.std(r) for gp in self.gps]
