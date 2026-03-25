import numpy as np
import jax.numpy as jnp


class PreferenceBasedLearning:

    def __init__(
        self,
        low: list[float] | np.ndarray,
        high: list[float] | np.ndarray,
        action_dims: list[float] | np.ndarray,
        preference_noise: float = 0.01,
        coactive_noise: float = 0.01,
        ordinal_noise: float = 0.01,
    ):
        """Sets up preference based learning based on POLAR (Tucker et al).
        This class is designed to organize the necessary likelihood functions
        and other variables that POLAR uses to learn preferences.

        Args:
            low: the lower bound of the action parameters
            high: the upper bound of the action parameters
            action_dims: the dimension of each action component (discretization)
            preference_noise: c_p, noisiness of preference feedback
            coactive_noise: c_c, noisiness of coactive feedback
            ordinal_noise: c_o, noisiness of ordinal feedback
        """
        # action space setup
        self.action_space = None
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.action_dims = np.asarray(action_dims, dtype=int)
        self.step = (self.high - self.low) / (self.action_dims - 1)
        self.generate_action_space()

        # feedback setup
        self.preference_noise = preference_noise
        self.coactive_noise = coactive_noise
        self.ordinal_noise = ordinal_noise
        self.preference_fbk = []
        self.coactive_fbk = []
        self.ordinal_fbk = []

        # compiled JAX arrays (set by compile())
        self._jax_pref = None
        self._jax_coac = None
        self._jax_ordi_idx = None
        self._jax_ordi_bounds = None
        self._compiled = False

    def add_feedback(
        self,
        preference: tuple[list[float] | np.ndarray, list[float] | np.ndarray, bool] | None,
        coactive: tuple[list[float] | np.ndarray, list[float] | np.ndarray, bool] | None,
        ordinal: tuple[list[float], float, float] | None,
    ) -> None:
        """Adds feedback to the PBL class in the form of (pairwise) preference,
        coactive, and ordinal.

        Args:
            preference: (curr, prev, p) where curr, prev are actions and p is
                        whether curr is preferred to prev
            coactive:   (a_bar, curr, c) where a_bar, curr are actions and c is
                        whether a_bar is preferred to curr
            ordinal:    (curr, b0, b1) where curr is an action and b0, b1 are a
                        bucket to fit r[a] to
        """
        if preference is not None:
            (curr, prev, p) = preference
            if not p:
                self.preference_fbk.append([self.get_idx(prev), self.get_idx(curr)])
            else:
                self.preference_fbk.append([self.get_idx(curr), self.get_idx(prev)])

        if coactive is not None:
            (a_bar, curr, c) = coactive
            if not c:
                self.coactive_fbk.append([self.get_idx(curr), self.get_idx(a_bar)])
            else:
                self.coactive_fbk.append([self.get_idx(a_bar), self.get_idx(curr)])

        if ordinal is not None:
            (curr, b0, b1) = ordinal
            self.ordinal_fbk.append([self.get_idx(curr), b0, b1])

        self._compiled = False

    def compile(self) -> None:
        """Compile feedback lists into JAX arrays for JIT-compatible likelihood
        computation. Must be called after adding feedback and before fitting
        the GP or using jax.grad/jax.jit on the likelihood functions.
        """
        self._jax_pref = None
        self._jax_coac = None
        self._jax_ordi_idx = None
        self._jax_ordi_bounds = None

        if self.preference_fbk:
            self._jax_pref = jnp.array(self.preference_fbk, dtype=jnp.int32)
        if self.coactive_fbk:
            self._jax_coac = jnp.array(self.coactive_fbk, dtype=jnp.int32)
        if self.ordinal_fbk:
            ordi = np.array(self.ordinal_fbk)
            self._jax_ordi_idx = jnp.array(ordi[:, 0], dtype=jnp.int32)
            self._jax_ordi_bounds = jnp.array(ordi[:, 1:])

        self._compiled = True

    def generate_action_space(self) -> None:
        """Generates an action space given initial parameters (see __init__)."""
        action_dims = []
        for _start, _stop, _num in zip(self.low, self.high, self.action_dims):
            action_dims.append(np.linspace(start=_start, stop=_stop, num=_num))

        # Generate the grid
        action_space = np.array(np.meshgrid(*action_dims))
        self.action_space = action_space.reshape(len(action_space), -1).T

    def get_idx(self, action) -> int:
        """Gets the closest index corresponding to a particular action in the
        generated action space.

        Args:
            action: the action to find the index of

        Returns:
            The index of the action in the discretized action space.
        """
        ret = 0
        for idx, (a, low, disc) in enumerate(zip(action, self.low, self.step)):
            ret += round((a - low) / disc) * np.prod(self.action_dims[:idx])
        return int(ret)

    @staticmethod
    def sigmoid(x):
        """Sigmoid function (JAX-compatible).

        Args:
            x: a numerical input to the sigmoid function

        Returns:
            The sigmoid of the input.
        """
        x = jnp.clip(x, -100, 100)
        return 1 / (1 + jnp.exp(-x))

    def pairwise_likelihood(self, r, a1, a2, noise=0):
        """Computes the negative log-likelihood of the user preferring action
        a1 over a2 (JAX-compatible).

        Args:
            r: user's latent reward function (discretized)
            a1: preferred action index (or array of indices)
            a2: non-preferred action index (or array of indices)
            noise: noise in the user's feedback

        Returns:
            Negative log-likelihood (scalar or array).
        """
        if noise != 0:
            ret = self.sigmoid((r[a1] - r[a2]) / noise)
        else:
            ret = jnp.where(r[a1] >= r[a2], 1.0, 0.01)
        return -jnp.log(ret)

    def preference_likelihood(self, r):
        """Negative log-likelihood of all preference feedback (JAX-compatible, vectorized)."""
        a1 = self._jax_pref[:, 0]
        a2 = self._jax_pref[:, 1]
        return jnp.sum(self.pairwise_likelihood(r, a1, a2, noise=self.preference_noise))

    def coactive_likelihood(self, r):
        """Negative log-likelihood of all coactive feedback (JAX-compatible, vectorized)."""
        a1 = self._jax_coac[:, 0]
        a2 = self._jax_coac[:, 1]
        return jnp.sum(self.pairwise_likelihood(r, a1, a2, noise=self.coactive_noise))

    def ordinal_likelihood(self, r):
        """Negative log-likelihood of all ordinal feedback (JAX-compatible, vectorized)."""
        a = self._jax_ordi_idx
        b0 = self._jax_ordi_bounds[:, 0]
        b1 = self._jax_ordi_bounds[:, 1]
        temp = self.sigmoid((b1 - r[a]) / self.ordinal_noise)
        temp = temp - self.sigmoid((b0 - r[a]) / self.ordinal_noise)
        return jnp.sum(-jnp.log(temp + 1e-8))

    def overall_likelihood(self, r):
        """Computes the overall negative log-likelihood across all feedback types.
        Compatible with jax.grad and jax.jit (call compile() first).

        The Python-level None checks are resolved at JAX trace time, so this
        function is safe to JIT as long as compile() has been called.
        """
        if not self._compiled:
            raise RuntimeError("Call compile() before using likelihood functions.")
        ret = 0.0
        if self._jax_pref is not None:
            ret = ret + self.preference_likelihood(r)
        if self._jax_coac is not None:
            ret = ret + self.coactive_likelihood(r)
        if self._jax_ordi_idx is not None:
            ret = ret + self.ordinal_likelihood(r)
        return ret

    def predict(self, r, a1, a2) -> bool:
        """Predicts whether the user prefers a1 over a2 given a latent reward function.

        Args:
            r: the learned latent reward function
            a1: first action
            a2: second action

        Returns:
            True if a1 is predicted to be preferred over a2.
        """
        return r[self.get_idx(a1)] >= r[self.get_idx(a2)]

    def optimal_action(self, r) -> np.ndarray:
        """Finds the optimal action given a latent reward function.

        Args:
            r: the learned latent reward function

        Returns:
            The action with the highest predicted reward.
        """
        return self.action_space[np.argmax(r)]
