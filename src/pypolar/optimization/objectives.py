"""Objective bookkeeping: raw measurements, their actions, and the affine
rescalings applied before they reach a GP."""

from dataclasses import asdict, dataclass
from functools import reduce

import numpy as np
import torch

from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples

BOUNDS_SLACK = 1e-9   # float round-off allowed outside a declared action box

def as_bounds(bounds, dim=None):
    """Any bounds spelling as the one form the package uses: a `(2, d)` array
    `[[low, ...], [high, ...]]`, the orientation BoTorch states bounds in.

    Args:
        bounds: a scalar, read as the half-width of the zero-centered box
            `[-bounds, bounds]`; a `(2,)` or `(2, 1)` pair, one low and one high
            shared by every dimension; or an already-`(2, d)` box.
        dim: d, broadcasting a shared pair across the action dimensions. None
            leaves the box at whatever width it was stated in.

    Returns:
        (2, d) low/high rows, in raw action units.
    """
    if np.ndim(bounds) == 0:
        bounds = [-bounds, bounds]

    box = np.asarray(bounds, dtype=float)
    if box.ndim == 1:
        box = box[:, None]
    if box.ndim != 2 or box.shape[0] != 2:
        raise ValueError('bounds must be (2, d) [[low, ...], [high, ...]], got '
                         f'shape {box.shape}')
    if dim is not None and box.shape[1] not in (1, dim):
        raise ValueError(f'bounds state {box.shape[1]} action dimensions, not {dim}')

    return box if dim is None else np.broadcast_to(box, (2, dim)).copy()


def sample_actions(
    bounds    : list | np.ndarray | float,
    n         : int,
    kind      : str,
    seed      : int,
    dim       : int = None
):
    """n actions over the box: Sobol is space-filling, uniform is iid.

    Args:
        bounds: the box, in any spelling `as_bounds` takes.
        dim: d, needed only when `bounds` states one low and one high for every
            dimension rather than a column per dimension.

    Returns:
        (n, d) actions.
    """
    box = as_bounds(bounds, dim)

    if kind == 'sobol':
        return draw_sobol_samples(bounds=torch.as_tensor(box, dtype=torch.float64),
                                  n=n, q=1, seed=seed).squeeze(1).numpy()

    if kind == 'uniform':
        return np.random.default_rng(seed).uniform(*box, size=(n, box.shape[1]))

    raise ValueError(f"kind must be 'sobol' or 'uniform', got {kind!r}")

@dataclass
class AffineTransform:
    scale: float
    shift: float

    def __post_init__(self):
        if np.any(self.scale == 0):
            raise ValueError('Scale cannot be zero')

    def __call__(self, data):
        return (data + self.shift) * self.scale

    def inv(self, data):
        return 1 / self.scale * data - self.shift

    def inv_scale(self, data):
        """Inverse for a spread: undoes the scaling but not the shift, since a
        shift does not move a standard deviation and a sign flip cannot make
        one negative."""
        return data / np.abs(self.scale)

    @classmethod
    def make_standardized(cls, data, sign=1.0, axis=None):
        std  = np.std(data, axis=axis)
        mean = np.mean(data, axis=axis)
        if np.any(std == 0):
            print('WARNING in AffineTransform.make_standardized: data has zero variance')

        return cls(
            scale = sign / (std + (std == 0)),   # zero-variance data is left unscaled
            shift = -mean
        )

    @classmethod
    def make_centered(cls, data, scale=1.0, axis=None):
        mean = np.mean(data, axis=axis)
        return cls(
            scale   = scale,
            shift   = -mean
        )

    @classmethod
    def make_normalized(cls, data, axis=None):
        span = np.max(data, axis) - np.min(data, axis)
        if np.any(span == 0):
            print('WARNING in AffineTransform.make_normalized: data has zero range')

        return cls(
            scale = 1 / (span + (span == 0)),   # zero-range data is left unscaled
            shift = -np.min(data, axis)
        )
        
    @classmethod
    def make_normalized_from_bounds(cls, low, high):
        """[low, high] onto [0, 1]. The same map as make_normalized, from a
        declared range rather than from whatever happens to be measured."""
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        span = high - low
        if np.any(span == 0):
            print('WARNING in AffineTransform.make_normalized_from_bounds: bounds have zero range')

        return cls(
            scale = 1 / (span + (span == 0)),   # zero-range bounds are left unscaled
            shift = -low
        )


@dataclass
class Objective:
    name:     str
    maximize: bool
    ydata:    np.ndarray        # (N,)
    xdata:    np.ndarray        # (N, K), where K is the action dimension
    column:   str | None = None # source dataframe column, when read from one
    action_bounds: np.ndarray | None = None # (2, K) actions, pins xtransform

    def __post_init__(self):
        self.ydata = np.asarray(self.ydata, dtype=float)
        self.xdata = np.asarray(self.xdata, dtype=float)
        if self.xdata.ndim == 1 and self.xdata.size:
            self.xdata = self.xdata[:, None]

        # declared bounds pin the [0, 1]^K frame so it does not move as points
        # arrive; without them it is the measured actions' own box
        if self.action_bounds is not None:
            # widened to K only once there are points to state K; an objective
            # declared before its first measurement holds the shared pair
            self.action_bounds = as_bounds(
                self.action_bounds,
                self.xdata.shape[1] if self.xdata.size else None
            )
            self._check_inside_bounds()
            self.xtransform = AffineTransform.make_normalized_from_bounds(*self.action_bounds)
        elif self.xdata.size:
            self.xtransform = AffineTransform.make_normalized(data=self.xdata, axis=0)

        if self.xdata.size:
            self.normalized_x = self.xtransform(self.xdata)

        if self.ydata.size:
            self.ytransform = AffineTransform.make_standardized(
                data = self.ydata,
                sign = self.sign
            )
            self.standard_y = self.ytransform(self.ydata)

    def _check_inside_bounds(self):
        """Actions must lie in the declared box. The slack is float round-off:
        an action optimized onto a boundary comes back a hair outside it."""
        if not self.xdata.size:
            return

        low, high = self.action_bounds
        slack   = BOUNDS_SLACK * (high - low)
        outside = (self.xdata < low - slack) | (self.xdata > high + slack)
        if np.any(outside):
            raise ValueError(f'{self.name}: actions '
                             f'{np.unique(np.nonzero(outside)[0]).tolist()} lie outside '
                             f'action_bounds {self.action_bounds}')

    def action_box(self):
        """(2, K) box over this objective's actions: action_bounds when they
        pin it, the measured actions' own range otherwise, None when empty."""
        if self.action_bounds is not None:
            return self.action_bounds
        if self.xdata.size:
            return np.stack([self.xdata.min(axis=0), self.xdata.max(axis=0)])

        return None

    @property
    def action_dim(self):
        """K, the action dimension."""
        return self.xdata.shape[1]

    @property
    def sign(self):
        return 1.0 if self.maximize else -1.0

    def best_action(self):
        # a positive rescale and shift do not move the argmax, and standard_y
        # is larger-is-better
        return self.xdata[np.argmax(self.standard_y)]

    def add_points(self, actions: np.ndarray, values: np.ndarray):
        actions = np.atleast_2d(actions)
        if len(self.xdata) == 0:
            # an empty objective has no action dimension until its first points
            self.xdata = np.empty((0, actions.shape[1]))

        previous = self.xdata, self.ydata
        self.xdata = np.vstack([self.xdata, actions])
        self.ydata = np.hstack([self.ydata, values])
        try:
            self.__post_init__()
        except ValueError:
            # a rejected point must not be left in the record
            self.xdata, self.ydata = previous
            raise
        
    def to_raw(self, mu, std=None):
        mu = self.ytransform.inv(mu)
        if std is None:
            return mu
        std = self.ytransform.inv_scale(std)
        return mu, std
    
    def to_record(self):
        """Everything this objective is, as plain data.
        """
        return asdict(self)

    @classmethod
    def from_record(cls, record):
        """The inverse of `to_record`.
        """
        return cls(**record)

    @classmethod
    def from_empty(cls, name, maximize, action_bounds=None):
        """An objective declared before any measurement.

        Args:
            action_bounds: (2, K) [[low, ...], [high, ...]] actions, in raw
                units, or any spelling `as_bounds` takes. Pins the [0, 1]^K
                frame so it does not move as points arrive, and measurements
                outside the box are rejected.
        """
        return cls(
            name          = name,
            maximize      = maximize,
            xdata         = np.array([]),
            ydata         = np.array([]),
            column        = None,
            action_bounds = action_bounds
        )

    @classmethod
    def from_df(cls, df, column, action_columns, maximize, name=None):
        return cls(
            name        = column if name is None else name,
            maximize    = maximize,
            ydata       = df[column].to_numpy(dtype=float),
            xdata       = df[action_columns].to_numpy(dtype=float),
            column      = column
        )
    
    @classmethod
    def from_data(cls, actions, values, maximize, name=None, action_bounds=None):
        """Measurements already in hand.

        Args:
            action_bounds: (2, K) [[low, ...], [high, ...]] actions, in raw
                units, or any spelling `as_bounds` takes. Pins the [0, 1]^K
                frame so it does not move as points arrive, and measurements
                outside the box are rejected.
        """
        return cls(
            name          = name,
            maximize      = maximize,
            ydata         = values,
            xdata         = actions,
            column        = None,
            action_bounds = action_bounds
        )
        
    @classmethod
    def from_synthetic(
        cls,
        function     : type[SyntheticTestFunction],
        actions      : np.ndarray,
        maximize     : bool,
        rel_noise_std: float,
        seed         : int,
        name         : str = None
    ):
        """Measurements of a BoTorch synthetic function, plus Gaussian noise.

        Args:
            function: a `SyntheticTestFunction` subclass, not an instance.
            actions: (N,) or (N, K) actions to evaluate at.
            maximize: whether larger values of the function are better.
            rel_noise_std: noise added, as a fraction of the truth's own spread.
            seed: seeds the noise draw.
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions[:, None]

        # bounds gate the domain: evaluate_true rejects actions outside them,
        # and custom bounds must contain a known optimizer, so span both
        dim = actions.shape[1]
        default = function(dim=dim).bounds.numpy()
        lo = np.minimum(actions.min(axis=0), default[0])
        hi = np.maximum(actions.max(axis=0), default[1])

        f = function(dim=dim, bounds=list(zip(lo, hi)))
        with torch.no_grad():
            y_true = f(torch.as_tensor(actions, dtype=torch.float64), noise=False).numpy()

        # noise is a fraction of the truth's own spread, so it means the same
        # thing across functions whose ranges differ by orders of magnitude
        noise_abs = rel_noise_std * y_true.std()
        y_noisy = y_true + noise_abs * np.random.default_rng(seed).standard_normal(y_true.shape)

        return cls.from_data(
            actions  = actions,
            values   = y_noisy,
            maximize = maximize,
            name     = name
        )


def _is_single(objs):
    """True when objs picks out one objective, by name or by position."""
    return isinstance(objs, (str, int, np.integer))


@dataclass
class DecoupledObjectives:
    """Objectives measured independently: each has its own actions and
    values, in any number."""
    objectives: list[Objective]

    def __post_init__(self):
        # one frame spanning every objective's box -- its action_bounds where
        # they are pinned and its measured actions otherwise -- so actions taken
        # from different objectives are normalized into the same box
        boxes = [box for box in (o.action_box() for o in self.objectives)
                 if box is not None]
        if not boxes:
            self.xtransform = None
            return

        # reduce rather than np.min(axis=0), so a box declared before its first
        # measurement -- still (2, 1) -- broadcasts against a per-dimension one
        self.xtransform = AffineTransform.make_normalized_from_bounds(
            low  = reduce(np.minimum, [box[0] for box in boxes]),
            high = reduce(np.maximum, [box[1] for box in boxes])
        )

    @property
    def names(self):
        return [o.name for o in self.objectives]

    @property
    def columns(self):
        return [o.column for o in self.objectives]

    @property
    def num_objectives(self):
        return len(self.objectives)

    # TODO: check over this later
    @property
    def action_dim(self):
        """K, the action dimension shared by every objective."""
        return self.objectives[0].action_dim

    @property
    def action_bounds(self):
        """(2, K) box spanning every objective's pinned bounds, or None when
        any of them is unpinned and so leaves the shared frame free to move."""
        bounds = [o.action_bounds for o in self.objectives]
        if not bounds or any(b is None for b in bounds):
            return None

        # reduce rather than np.min(axis=0), so a box declared before its first
        # measurement -- still (2, 1) -- broadcasts against a per-dimension one
        return np.stack([reduce(np.minimum, [b[0] for b in bounds]),
                         reduce(np.maximum, [b[1] for b in bounds])])

    def __len__(self):
        return len(self.objectives)

    def _one(self, obj):
        """The objective named or positioned by obj."""
        if isinstance(obj, str):
            return self.objectives[self.names.index(obj)]
        elif isinstance(obj, (int, np.integer)):
            return self.objectives[obj]
        else:
            raise TypeError('Objective selector must be a string or an int')

    def _select(self, objs=None):
        """The objectives picked out by objs, always as a list."""
        if objs is None:
            return list(self.objectives)
        elif isinstance(objs, slice):
            return self.objectives[objs]
        elif _is_single(objs):
            return [self._one(objs)]
        elif isinstance(objs, (list, tuple)):
            return [self._one(obj) for obj in objs]
        else:
            raise TypeError('DecoupledObjectives selector must be a string, an '
                            'int, a list of either, a slice or None')

    @staticmethod
    def _unwrap(values, objs):
        """Drops the list wrapper when a single objective was selected."""
        return values[0] if _is_single(objs) else values

    def __getitem__(self, objs):
        selected = self._select(objs)
        return selected[0] if _is_single(objs) else DecoupledObjectives(selected)

    def feedback(self, objs=None):
        """Standardized values of the selected objectives, each larger-is-better."""
        return self._unwrap([o.standard_y for o in self._select(objs)], objs)

    def actions(self, objs=None):
        """Actions of the selected objectives, in the shared normalized frame."""
        return self._unwrap([self.xtransform(o.xdata) for o in self._select(objs)], objs)

    def to_raw(self, mu, std=None, objs=None):
        """Posterior moments in maximization space, back in each objective's
        own units. Columns line up with the selected objectives.

        Args:
            mu: (n, m) posterior means.
            std: (n, m) posterior standard deviations, or None.
            objs: selector for the objectives the columns belong to.

        Returns:
            `mu` in raw units, or `(mu, std)` in raw units when `std` is given.
        """
        ts = [o.ytransform for o in self._select(objs)]
        mu = np.stack([t.inv(mu[..., i]) for i, t in enumerate(ts)], axis=-1)
        if std is None:
            return mu

        return mu, np.stack([t.inv_scale(std[..., i]) for i, t in enumerate(ts)],
                            axis=-1)

    def max(self, objs=None):
        """Largest standardized value of each selected objective."""
        return self._unwrap(
            np.array([o.standard_y.max() for o in self._select(objs)]), objs
        )

    def min(self, objs=None):
        """Smallest standardized value of each selected objective."""
        return self._unwrap(
            np.array([o.standard_y.min() for o in self._select(objs)]), objs
        )

    def range(self, objs=None):
        """Per-objective [min, max] of the standardized values. Shape (num_objs, 2),
        or (2,) for a single objective."""
        return np.stack([self.min(objs), self.max(objs)], axis=-1)

    @classmethod
    def from_df(cls, df, columns, action_columns, maximize, names=None):
        if isinstance(maximize, bool):
            maximize = [maximize] * len(columns)
        names = [None] * len(columns) if names is None else names
        return cls([Objective.from_df(df, c, action_columns, m, n)
                    for c, m, n in zip(columns, maximize, names)])

    @classmethod
    def from_empty(cls):
        return cls(objectives=[])

    def add_objective(self, objective: Objective):
        """Adds an entire objective 'axis'"""
        self.objectives.append(objective)
        self.__post_init__()

    def add_point(self, objs, actions: np.ndarray, values: np.ndarray):
        """Adds a single point to one or more objectives"""
        for obj in self._select(objs):
            obj.add_points(actions, values)

        self.__post_init__()