import numpy as np


class RandomSampler:
    """Selects new actions using uniform random sampling."""

    def sample(self, actions):
        """Sample a random action from the action space.

        Args:
            actions: array of available actions

        Returns:
            A single randomly selected action.
        """
        idx = np.random.choice(a=actions.shape[0], replace=False)
        return actions[idx]


class ThompsonSampler:
    """Selects actions via Thompson sampling from the GP posterior.

    Draws a sample from the posterior reward distribution and returns
    the action with the highest sampled reward. Falls back to random
    sampling when no GP has been fit yet.
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng()):
        """
        Args:
            gp: a BasicGP instance. Must have been fit (via gp.fit()) before
                Thompson sampling can be used. Before fitting, sample() falls
                back to uniform random.
        """
        self.gp = gp
        self.rng = rng
        self._posterior_cov = None
        self._posterior_L = None

    def update_posterior(self):
        """Recompute the posterior covariance from the GP's Hessian at the
        current MAP estimate. Call this after each gp.fit().

        The Laplace approximation gives posterior precision = Hessian of the
        negative log-posterior at the MAP. The posterior covariance is its inverse.
        """
        H = self.gp.hessian(self.gp.mu)
        self._posterior_cov = np.linalg.inv(H)
        # Cholesky for efficient sampling; regularize if needed
        eigvals = np.linalg.eigvalsh(self._posterior_cov)
        if eigvals.min() < 0:
            self._posterior_cov += (abs(eigvals.min()) + 1e-6) * np.eye(len(self.gp.mu))
        self._posterior_L = np.linalg.cholesky(self._posterior_cov)

    def sample(self, actions):
        """Sample an action by drawing from the GP posterior and picking the
        action with the highest sampled reward.

        Falls back to uniform random sampling if the GP hasn't been fit yet.

        Args:
            actions: array of available actions

        Returns:
            A single action from the action space.
        """
        if self.gp.mu is None or self._posterior_L is None:
            idx = self.rng.choice(a=actions.shape[0], replace=False)
            return actions[idx]

        # Draw a sample from the posterior: r ~ N(mu, posterior_cov)
        z = self.rng.standard_normal(len(self.gp.mu))
        r_sample = self.gp.mu + self._posterior_L @ z
        return actions[np.argmax(r_sample)]
