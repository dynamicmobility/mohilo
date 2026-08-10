import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pypolar.optimization.gp.base import GPModel


class ConjugateGP(GPModel):
    """Computes the exact GP posterior for regression problems. Feedback is 
    expected to be samples of a noisy function f(x) which the GP will fit to.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = None
        self.y = None
        self.sigma2 = None
        self.K_sX = None
        self._gram_chol = None

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
        M = idx.shape[0]

        X = action_space
        eps = self.JITTER
        # Cross/train blocks of the jittered prior covariance, without ever
        # materializing it: Sigma = K + eps*I, so eps lands only where indices
        # coincide.
        K_sX = self.kernel_cross(X, X[idx])                  # (N, M)
        K_XX = self.kernel_cross(X[idx], X[idx])             # (M, M)
        if M:
            K_sX[idx, np.arange(M)] += eps
            K_XX += eps * (idx[:, None] == idx[None, :])

        # Observation noise implied by the likelihood: L = lambda*||Sr-y||^2
        # is a Gaussian NLL with sigma^2 = 1/(2*lambda).
        self.sigma2 = 1.0 / (2.0 * precision)
        self.idx = idx
        self.y = y
        self.K_sX = K_sX
        self._gram_chol = (
            cho_factor(K_XX + self.sigma2 * np.eye(M)) if M else None
        )

    def fit(self, set_x0=True, **kwargs):
        """Compute the exact posterior mean ``mu = K_sX (K_XX + sigma^2 I)^-1 y``.

        Args:
            set_x0: accepted for interface compatibility; the closed form needs
                no warm start.
            **kwargs: accepted and ignored (e.g. scipy ``method``/``options``
                passed by callers that also drive the Laplace backend).

        Returns:
            The posterior mean vector.
        """
        if self._gram_chol is None:
            # No observations yet: the posterior mean is the (zero) prior mean.
            self.mu = np.zeros(self.actions.shape[0])
        else:
            self.mu = self.K_sX @ cho_solve(self._gram_chol, self.y)
        return self.mu

    def mu_at(self, X):
        """Posterior mean at arbitrary points, on or off the discretized grid.

        ``fit()`` evaluates the posterior at every action in ``action_space``;
        this is the continuous function underneath that vector,

            mu(x) = k(x, X_obs) (K_XX + sigma^2 I)^-1 y

        so no refit is involved — the Gram factor built by ``set_data`` is
        reused, and each call costs O(n M d) for n query points and M
        observations. On a grid point it reproduces the matching entry of ``mu``
        exactly, with one exception: at a *fed-back* action ``mu`` also carries
        the ``JITTER`` nugget that ``set_data`` adds where test and train indices
        coincide, so the two differ there by ``JITTER * alpha_m`` (~1e-5 times a
        weight). The jitter is a numerical device, not part of the model, so the
        continuous form deliberately omits it.

        Args:
            X: a single ``(d,)`` action or an ``(n, d)`` array of actions.

        Returns:
            Length-``n`` array of posterior means.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._gram_chol is None:
            # No observations yet: the posterior mean is the (zero) prior mean.
            return np.zeros(X.shape[0])
        return self.kernel_cross(X, self.actions[self.idx]) @ cho_solve(
            self._gram_chol, self.y
        )

    def std_at(self, X):
        """Posterior standard deviation at arbitrary points, on or off the grid.

        The continuous counterpart of ``std()``, the square root of

            sigma^2(x) = k(x, x) - k(x, X_obs) (K_XX + sigma^2 I)^-1 k(X_obs, x)

        reusing the Gram factor built by ``set_data``, so it costs O(n M^2) for n
        query points and involves no refit. Like ``mu_at`` it omits ``JITTER``:
        the nugget is a numerical device, not part of the model. ``std()`` does
        carry it in the prior variance, so the two differ by ``JITTER`` in
        variance (~5e-5 in standard deviation) at every action.

        Together with ``mu_at`` this makes any posterior-based acquisition a
        continuous function of the action, optimizable off grid.

        Args:
            X: a single ``(d,)`` action or an ``(n, d)`` array of actions.

        Returns:
            Length-``n`` array of standard deviations.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._gram_chol is None:
            # No observations yet: the posterior variance is the prior variance.
            return np.full(X.shape[0], np.sqrt(self.kernel.diag_var()))

        K_sX = self.kernel_cross(X, self.actions[self.idx])          # (n, M)
        var = self.kernel.diag_var() - np.einsum(
            "ij,ji->i", K_sX, cho_solve(self._gram_chol, K_sX.T)
        )
        return np.sqrt(np.clip(var, 0, None))

    def posterior_cov(self, r=None):
        """Full posterior covariance ``Sigma - K_sX G^-1 K_sX^T``.

        Materializing this is O(N^2) memory — prefer ``std()`` when only the
        diagonal is needed.

        Args:
            r: accepted for interface compatibility; the posterior is exact and
                does not depend on a linearization point.

        Returns:
            (N, N) posterior covariance matrix.
        """
        Sigma = self.prior_cov()
        if self._gram_chol is None:
            return Sigma
        return Sigma - self.K_sX @ cho_solve(self._gram_chol, self.K_sX.T)

    def posterior_cov_cross(self, idx):
        """Posterior covariance between every action and ``actions[idx]``, (N, C).

        Evaluated in data space, in O(N M C), without forming the N x N
        posterior covariance.

        Args:
            idx: length-C array of action indices.

        Returns:
            (N, C) block of the posterior covariance.
        """
        idx = np.asarray(idx, dtype=int)
        K_sc = self.kernel_cross(self.actions, self.actions[idx])
        K_sc[idx, np.arange(idx.shape[0])] += self.JITTER
        if self._gram_chol is None:
            return K_sc
        K_Xc = self.kernel_cross(self.actions[self.idx], self.actions[idx])
        K_Xc += self.JITTER * (self.idx[:, None] == idx[None, :])
        return K_sc - self.K_sX @ cho_solve(self._gram_chol, K_Xc)

    def std(self, r=None):
        """Per-action posterior standard deviation, in O(N M^2).

        Computes the variance diagonal directly rather than building the full
        N x N posterior covariance to take its diagonal.

        Args:
            r: accepted for interface compatibility; unused.

        Returns:
            Length-N array of standard deviations, one per action.
        """
        if self._gram_chol is None:
            return np.full(self.actions.shape[0], np.sqrt(self.prior_var()))
        V = cho_solve(self._gram_chol, self.K_sX.T)                  # (M, N)
        var = self.prior_var() - np.einsum("ij,ji->i", self.K_sX, V)
        return np.sqrt(np.clip(var, 0, None))

    def prepare_sampling(self):
        """Ensure the cached prior Cholesky exists.

        The data-dependent part of a draw is handled by Matheron's rule in
        ``sample_posterior()``, so this is free after the first call.
        """
        self._ensure_prior_chol()

    def sample_posterior(self, rng):
        """Draw one joint sample of the reward vector, via Matheron's rule:

            f_post = f_prior + K_sX G^-1 (y - f_prior[idx] - eps),

        with ``f_prior = L_Sigma z`` a draw from the *prior* and
        ``eps ~ N(0, sigma^2 I)`` simulated observation noise. This yields
        exactly the same distribution as ``mu + chol(Sigma_post) z`` but never
        forms or factorizes the N x N posterior covariance: the only
        decompositions involved are the cached prior Cholesky
        (data-independent) and the M x M Gram factor built by ``set_data()``.

        Args:
            rng: a numpy random Generator.

        Returns:
            Length-N sample of the reward vector.
        """
        L = self._ensure_prior_chol()
        f_prior = L @ rng.standard_normal(self.actions.shape[0])
        if self._gram_chol is None:
            # No observations yet: the posterior is the prior.
            return f_prior
        noise = np.sqrt(self.sigma2) * rng.standard_normal(self.idx.shape[0])
        resid = self.y - f_prior[self.idx] - noise
        return f_prior + self.K_sX @ cho_solve(self._gram_chol, resid)
