import numpy as np
import jax.numpy as jnp

class Likelihood:
    def __init__(
        self,
        low: list[float] | np.ndarray,
        high: list[float] | np.ndarray,
        action_dims: list[float] | np.ndarray,
    ):
        """Base class for likelihood functions used for PBL/HILO + GPs.

        Args:
            low: the lower bound of the action parameters
            high: the upper bound of the action parameters
            action_dims: the dimension of each action component (discretization)
        """
        # action space setup
        self.action_space = None
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.action_dims = np.asarray(action_dims, dtype=int)
        self.step = (self.high - self.low) / (self.action_dims - 1)
        self.generate_action_space()
        
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
    
class PreferenceBasedLearning(Likelihood):

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
        super().__init__(low, high, action_dims)

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

        # padded-array capacities for the data-parameterized fast path. These
        # grow geometrically so feedback_data() returns arrays whose shapes
        # change only occasionally, avoiding per-iteration JAX recompilation.
        self._cap = {"pref": 0, "coac": 0, "ordi": 0}

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

    @staticmethod
    def _grow_capacity(current: int, needed: int) -> int:
        """Return a capacity >= needed, growing geometrically from current.

        Capacity doubles whenever it is exceeded (like a dynamic array), so the
        number of distinct array shapes seen over a run is O(log n) rather than
        n. JAX recompiles only when a shape changes, so this bounds recompiles.
        """
        if needed <= current:
            return current
        cap = max(current, 1)
        while cap < needed:
            cap *= 2
        return cap

    def feedback_data(self) -> dict:
        """Build padded JAX feedback arrays for the data-parameterized fast path.

        Returns a pytree (dict) of padded index/mask arrays — one entry per
        feedback type that has data. Unused rows are masked out (mask=0) and
        point at index 0, so they contribute exactly zero to the likelihood,
        its gradient, and its Hessian. Array shapes are padded to a
        geometrically-growing capacity, so passing this dict as a runtime
        argument to a JIT-compiled likelihood avoids recompilation until a
        feedback type outgrows its current capacity.

        Pass the result to ``likelihood_from_data`` and to ``BasicGP.setup`` /
        ``BasicGP.set_feedback``.
        """
        data = {}

        if self.preference_fbk:
            n = len(self.preference_fbk)
            cap = self._cap["pref"] = self._grow_capacity(self._cap["pref"], n)
            idx = np.zeros((cap, 2), dtype=np.int32)
            idx[:n] = np.asarray(self.preference_fbk, dtype=np.int32)
            mask = np.zeros(cap)
            mask[:n] = 1.0
            data["pref"] = (jnp.array(idx), jnp.array(mask))

        if self.coactive_fbk:
            n = len(self.coactive_fbk)
            cap = self._cap["coac"] = self._grow_capacity(self._cap["coac"], n)
            idx = np.zeros((cap, 2), dtype=np.int32)
            idx[:n] = np.asarray(self.coactive_fbk, dtype=np.int32)
            mask = np.zeros(cap)
            mask[:n] = 1.0
            data["coac"] = (jnp.array(idx), jnp.array(mask))

        if self.ordinal_fbk:
            n = len(self.ordinal_fbk)
            cap = self._cap["ordi"] = self._grow_capacity(self._cap["ordi"], n)
            ordi = np.asarray(self.ordinal_fbk)
            a_idx = np.zeros(cap, dtype=np.int32)
            a_idx[:n] = ordi[:, 0]
            bounds = np.zeros((cap, 2))
            bounds[:n] = ordi[:, 1:]
            mask = np.zeros(cap)
            mask[:n] = 1.0
            data["ordi"] = (jnp.array(a_idx), jnp.array(bounds), jnp.array(mask))

        return data

    def _masked_pairwise(self, r, a1, a2, mask, noise):
        """Mask-weighted negative log-likelihood of pairwise feedback (a1 over a2)."""
        if noise != 0:
            nll = -jnp.log(self.sigmoid((r[a1] - r[a2]) / noise))
        else:
            nll = -jnp.log(jnp.where(r[a1] >= r[a2], 1.0, 0.01))
        return jnp.sum(mask * nll)

    def _masked_ordinal(self, r, a, b0, b1, mask):
        """Mask-weighted negative log-likelihood of ordinal feedback."""
        temp = self.sigmoid((b1 - r[a]) / self.ordinal_noise)
        temp = temp - self.sigmoid((b0 - r[a]) / self.ordinal_noise)
        return jnp.sum(mask * -jnp.log(temp + 1e-8))

    def likelihood_from_data(self, r, data):
        """Overall negative log-likelihood as a pure function of (r, data).

        ``data`` is a pytree produced by ``feedback_data()``. Because the
        feedback enters as a runtime argument (rather than a baked-in constant),
        a JIT-compiled version of this function is reused across iterations and
        recompiles only when an array shape changes. Compatible with jax.grad,
        jax.jit, and jax.hessian. The ``in data`` checks resolve at trace time.
        """
        ret = 0.0
        if "pref" in data:
            idx, mask = data["pref"]
            ret = ret + self._masked_pairwise(r, idx[:, 0], idx[:, 1], mask, self.preference_noise)
        if "coac" in data:
            idx, mask = data["coac"]
            ret = ret + self._masked_pairwise(r, idx[:, 0], idx[:, 1], mask, self.coactive_noise)
        if "ordi" in data:
            a_idx, bounds, mask = data["ordi"]
            ret = ret + self._masked_ordinal(r, a_idx, bounds[:, 0], bounds[:, 1], mask)
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


class Regression(Likelihood):
    
    def __init__(
        self,
        low: list[float] | np.ndarray,
        high: list[float] | np.ndarray,
        action_dims: list[float] | np.ndarray,
        precision: float = 1.0,
    ):
        """Sets up regression based learning, useful for HILO.

        Args:
            low: the lower bound of the action parameters
            high: the upper bound of the action parameters
            action_dims: the dimension of each action component (discretization)
        """
        super().__init__(low, high, action_dims)
        self.precision = precision
        self.feedback_data = np.array([], dtype=float).reshape(0, 2)  # shape (n, 2): [action_idx, value]
        
    def add_feedback(
        self, 
        action: list[float] | np.ndarray, 
        value: float
    ) -> None:
        """Adds regression feedback to the class.

        Args:
            action: the action corresponding to the observed value
            value: the observed value (regression target)
        """
        idx = self.get_idx(action)
        if self.feedback_data.size == 0:
            self.feedback_data = np.array([[idx, value]])
        else:
            self.feedback_data = np.vstack([self.feedback_data, [idx, value]])
        
    def likelihood(self, r):
        """Computes the negative log-likelihood of the regression data given a latent reward function.

        Args:
            r: the learned latent reward function
            y: the observed regression data
        """
        return self.precision * jnp.sum(
            (r[self.feedback_data[:, 0].astype(jnp.int32)] - self.feedback_data[:, 1]) ** 2
        )