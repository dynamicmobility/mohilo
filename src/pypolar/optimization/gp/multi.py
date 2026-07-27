import numpy as np

from pypolar.optimization.gp.botorch_gp import BoTorchGP
from pypolar.optimization.gp.conjugate import ConjugateGP
from pypolar.optimization.gp.laplace import LaplaceGP


class MultiObjectiveGP:
    """A collection of independent single-objective GPs sharing an action space.

    Each objective gets its own GP with its own kernel hyperparameters. The
    backend follows the feedback passed to ``setup()``: ``regressions`` selects
    the exact path named at construction (``ConjugateGP`` or ``BoTorchGP``),
    ``likelihoods`` the autodiff ``LaplaceGP`` path.
    """

    BACKENDS = {
        "conjugate": ConjugateGP,
        "botorch": BoTorchGP,
        "laplace": LaplaceGP,
    }

    # Backends that consume ``(idx, y, precision)`` regression feedback.
    REGRESSION_BACKENDS = ("conjugate", "botorch")

    def __init__(
        self,
        num_objs=1,
        kernels=['squared_exp'],
        signal_variances=[1],
        length_scales=[1],
        x0_init_methods=['random'],
        rng=np.random.default_rng(),
        backend="conjugate",
        fit_hypers=None,
        ard=None
    ):
        """
        Args:
            num_objs: number of objectives
            kernels: per-objective kernel name
            signal_variances: per-objective signal variance
            length_scales: per-objective length scale
            x0_init_methods: per-objective optimizer initialization (Laplace
                backend only)
            rng: numpy random Generator, shared by all objectives
            backend: which GP to build, ``'conjugate'``, ``'botorch'`` or
                ``'laplace'``. ``setup()`` switches between the regression and
                likelihood paths to match the feedback it is given; whichever of
                ``'conjugate'``/``'botorch'`` is named here is the one the
                regression path uses.
            fit_hypers: per-objective marginal-likelihood hyperparameter fitting
                (BoTorch backend only). None means off for every objective.
            ard: per-objective anisotropic length scales (BoTorch backend only).
                None means off for every objective.
        """
        self.num_objs = num_objs
        self.kernels = kernels
        self.signal_variances = signal_variances
        self.length_scales = length_scales
        self.x0_init_methods = x0_init_methods
        self.fit_hypers = fit_hypers or [False] * num_objs
        self.ard = ard or [False] * num_objs
        self.rng = rng
        self.regression_backend = self.BACKENDS[
            backend if backend in self.REGRESSION_BACKENDS else "conjugate"
        ]
        # Built eagerly so that `len(optimizer.gps)` and samplers constructed
        # with `gps=optimizer.gps` are valid before the first setup().
        self.gps: list = []
        self._build(self.BACKENDS[backend])

    def _build(self, backend):
        """Create one GP per objective using the given backend class.

        Mutates ``self.gps`` in place rather than reassigning it, so that a
        sampler holding a reference to the list still sees the models after a
        backend switch.
        """
        self.gps.clear()
        for i in range(self.num_objs):
            kwargs = dict(
                kernel=self.kernels[i],
                signal_variance=self.signal_variances[i],
                length_scale=self.length_scales[i],
                rng=self.rng,
            )
            if backend is LaplaceGP:
                kwargs["x0_init_method"] = self.x0_init_methods[i]
            if backend is BoTorchGP:
                kwargs["fit_hypers"] = self.fit_hypers[i]
                kwargs["ard"] = self.ard[i]
            self.gps.append(backend(**kwargs))

    def setup(self, action_space, likelihoods=None, regressions=None):
        """Set up each per-objective GP.

        Args:
            action_space: (N, d) array of discretized actions.
            likelihoods: per-objective JAX-compatible likelihood functions,
                selecting the ``LaplaceGP`` backend.
            regressions: per-objective ``(idx, y, precision)`` tuples (e.g. from
                ``MultiObjectiveRegression.get_regression_data()``), selecting
                the exact backend named at construction. Preferred when
                applicable — it never builds an N x N matrix.
        """
        if (likelihoods is None) == (regressions is None):
            raise ValueError(
                "Pass exactly one of `likelihoods` or `regressions`."
            )

        backend = self.regression_backend if regressions is not None else LaplaceGP
        if type(self.gps[0]) is not backend:
            self._build(backend)

        for i, gp in enumerate(self.gps):
            if regressions is not None:
                gp.set_data(action_space, *regressions[i])
            else:
                gp.set_data(action_space, likelihoods[i])

    def fit(self, set_x0=True, **kwargs):
        """Fit every objective. Returns the list of posterior means."""
        for gp in self.gps:
            gp.fit(set_x0=set_x0, **kwargs)

        return [gp.mu for gp in self.gps]

    def std(self, r=None):
        """Per-objective posterior standard deviations."""
        return [gp.std(r) for gp in self.gps]
