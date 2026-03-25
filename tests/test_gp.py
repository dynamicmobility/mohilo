import numpy as np
import jax.numpy as jnp
import pytest
from pypolar import BasicGP


class TestKernel:
    def test_squared_exp_kernel_diagonal(self):
        gp = BasicGP(signal_var=1.0, lengthscale=1.0)
        X = np.array([[0], [1], [2]], dtype=float)
        gp.actions = X
        K = gp.squared_exp_kernel(X)
        np.testing.assert_allclose(np.diag(K), np.ones(3))

    def test_squared_exp_kernel_symmetry(self):
        gp = BasicGP(signal_var=1.0, lengthscale=1.0)
        X = np.random.rand(5, 2)
        gp.actions = X
        K = gp.squared_exp_kernel(X)
        np.testing.assert_allclose(K, K.T)

    def test_squared_exp_kernel_positive_definite(self):
        gp = BasicGP(signal_var=1.0, lengthscale=1.0)
        X = np.random.rand(10, 2)
        gp.actions = X
        K = gp.squared_exp_kernel(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10)


class TestSetup:
    def test_setup_initializes_mu(self):
        gp = BasicGP()
        X = np.random.rand(10, 2)
        gp.setup(X, likelihood=lambda r: 0.0)
        assert gp.mu is not None
        assert gp.mu.shape == (10,)

    def test_setup_cov_shape(self):
        gp = BasicGP()
        X = np.random.rand(10, 2)
        gp.setup(X, likelihood=lambda r: 0.0)
        assert gp.cov.shape == (10, 10)

    def test_jacobian_always_available(self):
        gp = BasicGP()
        X = np.random.rand(5, 2)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        assert gp.jacobian is not None
        jac = gp.jacobian(np.random.rand(5))
        assert jac.shape == (5,)

    def test_hessian_always_available(self):
        gp = BasicGP()
        X = np.random.rand(5, 2)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        assert gp.hessian is not None
        hess = gp.hessian(np.random.rand(5))
        assert hess.shape == (5, 5)


class TestFit:
    def test_fit_runs(self):
        gp = BasicGP(lengthscale=1, signal_var=1)
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        mu = gp.fit(method='L-BFGS-B')
        assert mu.shape == (5,)

    def test_fit_reduces_objective(self):
        np.random.seed(42)
        gp = BasicGP(lengthscale=1, signal_var=1)
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        obj_before = gp.objective(gp.mu)
        gp.fit(method='L-BFGS-B')
        obj_after = gp.objective(gp.mu)
        assert obj_after <= obj_before
