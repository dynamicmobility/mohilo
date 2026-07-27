import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from scipy.stats import norm, qmc
import pypolar as plr


def expected_max_of_lines(a, b):
    """``E_Z[max_i (a_i + b_i Z)]`` for ``Z ~ N(0, 1)``.

    The maximum over ``i`` is the upper envelope of a set of lines in ``Z``, so
    the expectation is exact: keep only the lines that attain the envelope
    somewhere, and integrate each over the interval of ``Z`` on which it is on
    top.

    Args:
        a: length-n array of intercepts.
        b: length-n array of slopes.

    Returns:
        The expectation, a float.
    """
    order = np.lexsort((a, b))
    a, b = a[order], b[order]

    # Among equal slopes only the largest intercept can ever be on top.
    keep = np.append(np.diff(b) > 0, True)
    a, b = a[keep], b[keep]

    # Sweep in order of increasing slope, popping lines whose crossing point
    # with the incoming line falls below the previous crossing.
    idx = [0]
    z = [-np.inf]
    for i in range(1, len(b)):
        while True:
            crossing = (a[idx[-1]] - a[i]) / (b[i] - b[idx[-1]])
            if crossing > z[-1]:
                break
            idx.pop()
            z.pop()
        idx.append(i)
        z.append(crossing)

    a, b = a[idx], b[idx]
    z = np.append(z, np.inf)
    cdf, pdf = norm.cdf(z), norm.pdf(z)
    return float(np.sum(a * np.diff(cdf) + b * -np.diff(pdf)))


def _observation_noise(gp, noise_var):
    """``noise_var``, or the GP's own observation noise variance."""
    if noise_var is not None:
        return noise_var
    sigma2 = getattr(gp, "sigma2", None)
    if sigma2 is None:
        raise AttributeError(
            f"{type(gp).__name__} does not expose an observation noise variance; "
            "pass noise_var explicitly."
        )
    return sigma2


def knowledge_gradient(mu, sigma_tilde):
    """Expected increase in ``max(mu)`` from one observation.

    Args:
        mu: length-n posterior mean.
        sigma_tilde: length-n direction the mean moves in, per unit standard
            normal deviate of the observation.

    Returns:
        A non-negative float.
    """
    return max(expected_max_of_lines(mu, sigma_tilde) - mu.max(), 0.0)


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


class AcquisitionSampler:
    """Base for samplers that score every action and take the argmax.

    Subclasses implement ``acquisition(actions)``. Until the GP has been fit and
    ``update_posterior()`` called -- and for the first ``n_warmup`` queries --
    ``sample()`` falls back to uniform random.
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng(),
                 n_warmup: int = 1):
        """
        Args:
            gp: a GPModel instance.
            rng: a numpy random number generator.
            n_warmup: number of leading queries drawn uniformly at random.
        """
        self.gp = gp
        self.rng = rng
        self.n_warmup = n_warmup
        self._ready = False
        self._num_queries = 0

    def update_posterior(self):
        """Mark the posterior as usable. Call this after each gp.fit()."""
        self._ready = True

    def acquisition(self, actions):
        """Score every action. Higher is better.

        Args:
            actions: array of available actions, shape (N, d).

        Returns:
            Length-N array of scores.
        """
        raise NotImplementedError

    def sample(self, actions):
        """Return the action with the highest acquisition score.

        Ties are broken uniformly at random.

        Args:
            actions: array of available actions, shape (N, d).

        Returns:
            A single action from the action space.
        """
        self._num_queries += 1
        if (self.gp.mu is None or not self._ready
                or self._num_queries <= self.n_warmup):
            return actions[self.rng.choice(actions.shape[0])]

        scores = self.acquisition(actions)
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        return actions[self.rng.choice(best)]


class ExpectedImprovementSampler(AcquisitionSampler):
    """Selects actions by expected improvement over the best posterior mean.

        EI(x) = s(x) * (z Phi(z) + phi(z)),   z = (mu(x) - max(mu) - xi) / s(x)

    The incumbent is the best posterior *mean* rather than the best observed
    value, which under noisy observations is biased upward. The same action can
    be returned more than once; repeat observations average the noise down.
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng(),
                 xi: float = 0.0, n_warmup: int = 1):
        """
        Args:
            gp: a GPModel instance.
            rng: a numpy random number generator.
            xi: improvement threshold. Larger values explore more.
            n_warmup: number of leading queries drawn uniformly at random.
        """
        super().__init__(gp, rng, n_warmup)
        self.xi = xi

    def acquisition(self, actions):
        mu, std = self.gp.mu, self.gp.std()
        safe = np.where(std > 0, std, 1.0)
        z = (mu - mu.max() - self.xi) / safe
        return np.where(std > 0, std * (z * norm.cdf(z) + norm.pdf(z)), 0.0)


class KnowledgeGradientSampler(AcquisitionSampler):
    """Selects actions by the expected increase in the best posterior mean.

    Observing action ``c`` moves the whole posterior mean along a fixed
    direction scaled by one standard normal deviate,

        sigma_tilde = Sigma[:, c] / sqrt(noise_var + Sigma[c, c]),

    so ``KG(c) = E_Z[max(mu + sigma_tilde Z)] - max(mu)`` is available in closed
    form. Costs O(C N log N) for C candidates over N actions. The same action
    can be returned more than once.
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng(),
                 noise_var: float = None, num_candidates: int = None,
                 n_warmup: int = 1):
        """
        Args:
            gp: a GPModel instance.
            rng: a numpy random number generator.
            noise_var: observation noise variance. Defaults to the GP's own
                ``sigma2``, which only the ConjugateGP backend defines.
            num_candidates: number of actions to score, drawn uniformly without
                replacement. None scores every action.
            n_warmup: number of leading queries drawn uniformly at random.
        """
        super().__init__(gp, rng, n_warmup)
        self.noise_var = noise_var
        self.num_candidates = num_candidates

    def acquisition(self, actions):
        n = actions.shape[0]
        if self.num_candidates is None or self.num_candidates >= n:
            idx = np.arange(n)
        else:
            idx = self.rng.choice(n, size=self.num_candidates, replace=False)

        mu = self.gp.mu
        cov = self.gp.posterior_cov_cross(idx)                       # (N, C)
        noise_var = _observation_noise(self.gp, self.noise_var)
        denom = np.sqrt(noise_var + cov[idx, np.arange(idx.shape[0])])

        scores = np.full(n, -np.inf)
        for j, c in enumerate(idx):
            scores[c] = knowledge_gradient(mu, cov[:, j] / denom[j])
        return scores


class MaxValueEntropySampler(AcquisitionSampler):
    """Selects actions by mutual information with the maximum reward value.

    The maxima of joint posterior samples serve as samples of ``f*``, and the
    information gain about ``f*`` from observing ``x`` has the closed form

        g   = (f* - mu(x)) / sqrt(s(x)^2 + noise_var)
        MES = mean over f* of [ g phi(g) / (2 Phi(g)) - log Phi(g) ].
    """

    def __init__(self, gp, rng: np.random.Generator = np.random.default_rng(),
                 num_maxima: int = 32, noise_var: float = None,
                 n_warmup: int = 1):
        """
        Args:
            gp: a GPModel instance.
            rng: a numpy random number generator.
            num_maxima: number of posterior samples of ``f*``.
            noise_var: observation noise variance. Defaults to the GP's own
                ``sigma2``, which only the ConjugateGP backend defines.
            n_warmup: number of leading queries drawn uniformly at random.
        """
        super().__init__(gp, rng, n_warmup)
        self.num_maxima = num_maxima
        self.noise_var = noise_var
        self._maxima = None

    def update_posterior(self):
        """Redraw the samples of ``f*``. Call this after each gp.fit()."""
        self.gp.prepare_sampling()
        self._maxima = np.array([
            self.gp.sample_posterior(self.rng).max()
            for _ in range(self.num_maxima)
        ])
        self._ready = True

    def acquisition(self, actions):
        mu = self.gp.mu
        noise_var = _observation_noise(self.gp, self.noise_var)
        denom = np.sqrt(self.gp.std() ** 2 + noise_var)
        maxima = np.maximum(self._maxima, mu.max())

        g = (maxima[:, None] - mu[None, :]) / denom[None, :]
        cdf = np.clip(norm.cdf(g), 1e-12, None)
        return (g * norm.pdf(g) / (2 * cdf) - np.log(cdf)).mean(axis=0)


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


class QNEHVISampler:
    """Selects actions by noisy expected hypervolume improvement, multi-objective.

    Scores each action by how much it is expected to grow the hypervolume
    dominated by the Pareto front, under the joint posterior over all
    objectives and marginalizing over the noise in the observations collected so
    far. Unlike ``DSTSampler`` this reasons about the front directly rather than
    about one scalarization at a time.

    The acquisition is BoTorch's ``qLogNoisyExpectedHypervolumeImprovement``
    evaluated by enumeration over the discrete action space, so it needs
    ``BoTorchGP`` models -- the other backends expose no BoTorch model to build
    it from.
    """

    def __init__(
        self,
        gps: list,
        rng: np.random.Generator = np.random.default_rng(),
        ref_point=None,
        num_samples: int = 128,
        max_batch_size: int = 1024,
        n_warmup: int = 1
    ):
        """
        Args:
            gps: a list of ``BoTorchGP`` instances, one per objective. Must have
                been fit before ``update_posterior()`` is called; until then
                ``sample()`` falls back to uniform random.
            rng: a numpy random number generator.
            ref_point: per-objective reward below which an outcome contributes
                no hypervolume. None infers it from the feedback collected so
                far, which is the usual choice when the objective scale is not
                known up front.
            num_samples: number of quasi-Monte-Carlo samples used to estimate
                the acquisition.
            max_batch_size: number of actions scored per forward pass. Bounds
                peak memory on large action spaces.
            n_warmup: number of leading queries drawn uniformly at random.
        """
        self.gps = gps
        self.rng = rng
        self.ref_point = ref_point
        self.num_samples = num_samples
        self.max_batch_size = max_batch_size
        self.n_warmup = n_warmup
        self._acqf = None
        self._num_queries = 0

    def _reference_point(self):
        """Per-objective reference point, as a list."""
        if self.ref_point is not None:
            return list(np.asarray(self.ref_point, dtype=float).ravel())
        observed = torch.as_tensor(
            np.stack([gp.y for gp in self.gps], axis=-1), dtype=torch.float64
        )
        return infer_reference_point(observed).tolist()

    def update_posterior(self):
        """Rebuild the acquisition around the current posteriors.

        Call this after each ``fit()``: the acquisition holds the models and the
        set of already-queried actions, both of which change every iteration.
        """
        missing = [gp for gp in self.gps if getattr(gp, "model", None) is None]
        if missing or self.gps[0].mu is None:
            self._acqf = None
            return

        anchor = self.gps[0]
        self._acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=ModelListGP(*[gp.model for gp in self.gps]),
            ref_point=self._reference_point(),
            X_baseline=torch.as_tensor(
                anchor.actions[anchor.idx], dtype=torch.float64
            ),
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_samples]),
                seed=int(self.rng.integers(2 ** 32)),
            ),
            prune_baseline=True,
        )

    def sample(self, actions):
        """Return the action with the highest acquisition score.

        Falls back to uniform random sampling before the GPs have been fit and
        for the first ``n_warmup`` queries.

        Args:
            actions: array of available actions, shape (N, d).

        Returns:
            A single action from the action space.
        """
        self._num_queries += 1
        if self._acqf is None or self._num_queries <= self.n_warmup:
            return actions[self.rng.choice(actions.shape[0])]

        with torch.no_grad():
            candidate, _ = optimize_acqf_discrete(
                self._acqf,
                q=1,
                choices=torch.as_tensor(actions, dtype=torch.float64),
                max_batch_size=self.max_batch_size,
            )
        return candidate.reshape(-1).numpy()