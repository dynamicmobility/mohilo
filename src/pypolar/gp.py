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
        signal_var=1,
        lengthscale=1,
        mu_init_method="random",
    ):
        """Approximates an unknown latent reward function using a Gaussian
        process using a provided objective function.

        Args:
            kernel: kernel type (currently only 'squared_exp')
            signal_var: signal variance (dictates expected variation in the latent reward)
            lengthscale: lengthscale (dictates the smoothness of the latent reward)
            mu_init_method: initialization method for the mean ('random')
        """
        self.signal_var = signal_var
        self.lengthscale = lengthscale
        self.mu_init_method = mu_init_method
        self.mu = None
        self.f = None
        if kernel == "squared_exp":
            self.kernel = self.squared_exp_kernel

    def setup(self, action_space, likelihood):
        """Initialize the GP with an action space and likelihood function.

        Automatically derives the Jacobian and Hessian of the full objective
        (likelihood + GP prior) using JAX autodiff. All three are JIT-compiled
        and wrapped for scipy compatibility.

        Args:
            action_space: array of discretized actions
            likelihood: negative log-likelihood function from PBL (must be
                        JAX-compatible, i.e. call pbl.compile() first)
        """
        self.actions = action_space
        self.cov = self.prior_cov()
        np.fill_diagonal(self.cov, np.diagonal(self.cov) + 1e-5)
        self.cov_inv = np.linalg.inv(self.cov)
        if self.mu_init_method == "random":
            self.mu = 2 * np.random.random(self.actions.shape[0]) - 1
        else:
            raise ValueError(f"Invalid mu init method: {self.mu_init_method}")

        self.likelihood = likelihood

        # JAX-compatible objective: likelihood + GP prior
        jax_cov_inv = jnp.array(self.cov_inv)

        def _objective(r):
            return likelihood(r) + 0.5 * r @ jax_cov_inv @ r

        # JIT-compile objective, gradient, and Hessian
        self._jax_objective = jax.jit(_objective)
        self._jax_jacobian = jax.jit(jax.grad(_objective))
        self._jax_hessian = jax.jit(jax.hessian(_objective))

        # Scipy-compatible wrappers (numpy in, numpy out)
        self.objective = lambda r: float(self._jax_objective(jnp.asarray(r)))
        self.jacobian = lambda r: np.asarray(self._jax_jacobian(jnp.asarray(r)))
        self.hessian = lambda r: np.asarray(self._jax_hessian(jnp.asarray(r)))

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
        K = self.signal_var**2 * np.exp(-0.5 * sqdist / self.lengthscale**2)
        return K

    def prior_cov(self):
        """Computes the prior covariance matrix using the kernel."""
        return self.kernel(self.actions)

    def fit(self, **kwargs):
        """Fits the Gaussian process to the user feedback.

        Gradients and Hessians are always available via JAX autodiff.

        Args:
            **kwargs: passed to scipy.optimize.minimize (e.g. method, options)

        Returns:
            The optimized mean vector.
        """
        res = minimize(
            self.objective,
            self.mu,
            jac=self.jacobian,
            hess=self.hessian,
            **kwargs,
        )
        self.mu = res.x
        return self.mu

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
