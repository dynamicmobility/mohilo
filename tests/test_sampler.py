import numpy as np
import jax.numpy as jnp
from pypolar import RandomSampler, ThompsonSampler, BasicGP


class TestRandomSampler:
    def test_sample_returns_action(self):
        sampler = RandomSampler()
        actions = np.array([[0, 0], [1, 1], [2, 2]])
        action = sampler.sample(actions)
        assert action.shape == (2,)

    def test_sample_is_from_action_space(self):
        sampler = RandomSampler()
        actions = np.array([[0], [1], [2], [3]])
        for _ in range(20):
            action = sampler.sample(actions)
            assert action in actions


class TestThompsonSampler:
    def test_fallback_to_random_before_fit(self):
        gp = BasicGP()
        sampler = ThompsonSampler(gp)
        actions = np.array([[0], [1], [2], [3]])
        action = sampler.sample(actions)
        assert action in actions

    def test_sample_after_fit(self):
        np.random.seed(42)
        gp = BasicGP(lengthscale=1, signal_var=1)
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        gp.fit(method='L-BFGS-B')
        sampler = ThompsonSampler(gp)
        sampler.update_posterior()
        action = sampler.sample(X)
        assert action.shape == (1,)
        assert action in X

    def test_posterior_cov_set_after_update(self):
        np.random.seed(42)
        gp = BasicGP(lengthscale=1, signal_var=1)
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        gp.fit(method='L-BFGS-B')
        sampler = ThompsonSampler(gp)
        assert sampler._posterior_L is None
        sampler.update_posterior()
        assert sampler._posterior_L is not None
        assert sampler._posterior_L.shape == (5, 5)

    def test_samples_vary(self):
        np.random.seed(0)
        gp = BasicGP(lengthscale=1, signal_var=1)
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        gp.setup(X, likelihood=lambda r: jnp.sum(r**2))
        gp.fit(method='L-BFGS-B')
        sampler = ThompsonSampler(gp)
        sampler.update_posterior()
        samples = [sampler.sample(X).item() for _ in range(50)]
        assert len(set(samples)) > 1
