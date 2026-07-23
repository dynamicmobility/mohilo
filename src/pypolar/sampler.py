import numpy as np
from scipy.stats import qmc
import pypolar as plr


class RandomSampler:
    """Selects new actions using uniform random sampling."""
    def __init__(self, rng: np.random.Generator = np.random.default_rng()):
        """
        Args:
            rng: a numpy random number generator. Used for reproducibility.
        """
        self.rng = rng

    def sample(self, actions):
        """Sample a random action from the action space.

        Args:
            actions: array of available actions

        Returns:
            A single randomly selected action.
        """
        idx = self.rng.choice(a=actions.shape[0], replace=False)
        return actions[idx]
    
    def update_posterior(self):
        pass
    
class UniformSampler:
    """Quasi-uniformly covers the action space using a Sobol sequence.

    Given the total number of queries ``n``, a low-discrepancy Sobol sequence
    is generated over the Nd bounding box of the action space and each point is
    snapped to the nearest available discrete action. Compared to independent
    uniform sampling, this spreads the queries far more evenly across the space
    (lower discrepancy), which is desirable for space-filling initial designs.
    """

    def __init__(self, n: int, rng: np.random.Generator = np.random.default_rng(),
                 scramble: bool = True, no_repeat: bool = True):
        """
        Args:
            n: total number of queries the sampler will be asked to produce.
                Used to size the Sobol sequence. Sobol's balance properties hold
                best at powers of two, so ``2**ceil(log2(n))`` points are drawn
                and the first ``n`` are used.
            rng: a numpy random number generator. Used to seed the (optional)
                Sobol scrambling for reproducibility.
            scramble: whether to apply Owen scrambling to the Sobol sequence.
                Scrambling breaks up the deterministic structure while keeping
                low discrepancy, and is generally recommended.
            no_repeat: if True, never return the same discrete action twice.
                When a Sobol point snaps to an already-used action, the nearest
                unused action is chosen instead.
        """
        self.n = int(n)
        self.rng = rng
        self.scramble = scramble
        self.no_repeat = no_repeat
        self._points = None   # (n, d) Sobol points scaled to the action bounds
        self._cursor = 0
        self._used = set()

    def _init_points(self, actions):
        """Lazily build the Sobol point cloud once the action dimension is known.

        Args:
            actions: array of available actions, shape (N, d).
        """
        d = actions.shape[1]
        low = actions.min(axis=0)
        high = actions.max(axis=0)

        engine = qmc.Sobol(d=d, scramble=self.scramble, seed=self.rng)
        # Draw a power-of-two number of points to preserve Sobol balance, then
        # keep only the n we actually need.
        m = int(np.ceil(np.log2(self.n))) if self.n > 1 else 0
        unit = engine.random_base2(m)[:self.n]        # (n, d) in [0, 1]^d
        self._points = qmc.scale(unit, low, high)     # scaled to the box

    def sample(self, actions):
        """Return the next Sobol point snapped to the nearest available action.

        Args:
            actions: array of available actions, shape (N, d).

        Returns:
            A single action from the action space.
        """
        if self._points is None:
            self._init_points(actions)

        # Wrap around if asked for more than n samples (falls back to reusing
        # the sequence; only matters if the caller exceeds the declared n).
        point = self._points[self._cursor % len(self._points)]
        self._cursor += 1

        # Distance from the Sobol point to every discrete action.
        dists = np.linalg.norm(actions - point, axis=1)
        if self.no_repeat and self._used:
            dists = dists.copy()
            dists[list(self._used)] = np.inf
        idx = int(np.argmin(dists))
        self._used.add(idx)
        return actions[idx]

    def update_posterior(self):
        pass


class ThompsonSampler:
    """Selects actions via Thompson sampling from the GP posterior.

    Draws a sample from the posterior reward distribution and returns
    the action with the highest sampled reward. Falls back to random
    sampling when no GP has been fit yet.
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng()):
        """
        Args:
            gp: a GPModel instance (ConjugateGP or LaplaceGP). Must have been fit (via gp.fit()) before
                Thompson sampling can be used. Before fitting, sample() falls
                back to uniform random.
        """
        self.gp = gp
        self.rng = rng
        self._ready = False

    def update_posterior(self):
        """Prepare the GP for posterior sampling. Call this after each gp.fit().

        On the regression (Woodbury) path this is essentially free after the
        first call; on the Laplace path it refactorizes the posterior covariance,
        whose Hessian at the MAP is the posterior precision.
        """
        self.gp.prepare_sampling()
        self._ready = True

    def sample(self, actions):
        """Sample an action by drawing from the GP posterior and picking the
        action with the highest sampled reward.

        Falls back to uniform random sampling if the GP hasn't been fit yet.

        Args:
            actions: array of available actions

        Returns:
            A single action from the action space.
        """
        if self.gp.mu is None or not self._ready:
            idx = self.rng.choice(a=actions.shape[0], replace=False)
            return actions[idx]

        # Draw a sample from the posterior: r ~ N(mu, posterior_cov)
        r_sample = self.gp.sample_posterior(self.rng)
        return actions[np.argmax(r_sample)]


class DSTSampler:
    def __init__(
        self, 
        gps: list[plr.GPModel],
        rng: np.random.Generator = np.random.default_rng(),
        rho=0.05
    ):
        """
        Dueling Scalarized Thompson Sampling (DST) as a multi-objective 
        acquisition function.
        Args:
            gps: a list of GPModel instances. Must have been fit (via gp.fit()) before
                Thompson sampling can be used. Before fitting, sample() falls
                back to uniform random.
        """
        self.gps = gps
        self.rng = rng
        self._ready = False
        self.rho = rho

    def update_posterior(self):
        """Prepare each GP for posterior sampling. Call this after each gp.fit().
        """
        for gp in self.gps:
            gp.prepare_sampling()
        self._ready = True

    def sample(self, actions):
        """Sample an action by drawing from the GP posterior and picking the
        action with the highest sampled scalarized reward w.r.t. a random weight
        vector drawn uniformly from the simplex.
        
        Falls back to uniform random sampling if the GPs haven't been fit yet.
        
        Args:
            actions: array of available actions

        Returns:
            A single action from the action space.
        """
        if self.gps[0].mu is None or not self._ready:
            idx = self.rng.choice(a=actions.shape[0], replace=False)
            return actions[idx]

        # One independent posterior draw per objective
        f_sample = np.array([gp.sample_posterior(self.rng) for gp in self.gps])

        # Normalize objectives
        lo = f_sample.min(axis=1, keepdims=True)
        hi = f_sample.max(axis=1, keepdims=True)
        rng_ = hi - lo
        flat = rng_ <= 1e-12
        f_norm = np.where(flat, 0.0, (f_sample - lo) / np.where(flat, 1.0, rng_))

        # Draw a sample
        theta = self.rng.dirichlet(alpha=np.ones(len(self.gps)))
        weighted = f_norm.T * theta
        r_sample = weighted.min(axis=1) + self.rho * weighted.sum(axis=1)

        return actions[np.argmax(r_sample)]