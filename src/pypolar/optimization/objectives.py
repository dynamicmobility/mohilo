"""Objective bookkeeping: raw measurements, their actions, and the affine
rescalings applied before they reach a GP."""

from dataclasses import dataclass

import numpy as np


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

    @classmethod
    def make_standardized(cls, data, axis=None):
        std  = np.std(data, axis=axis)
        mean = np.mean(data, axis=axis)
        if np.any(std == 0):
            print('WARNING in AffineTransform.make_standardized: data has zero variance')

        return cls(
            scale = 1 / (std + (std == 0)),   # zero-variance data is left unscaled
            shift = -mean
        )

    @classmethod
    def make_centered(cls, data, axis=None):
        mean = np.mean(data, axis=axis)
        return cls(
            scale   = 1.0,
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


@dataclass
class Objective:
    name:     str
    maximize: bool
    ydata:    np.ndarray        # (N,)
    xdata:    np.ndarray        # (N, K), where K is the action dimension
    column:   str | None = None # source dataframe column, when read from one

    def __post_init__(self):
        self.ydata = np.asarray(self.ydata, dtype=float)
        self.xdata = np.asarray(self.xdata, dtype=float)
        if self.xdata.ndim == 1:
            self.xdata = self.xdata[:, None]

        # scale = sign, so the transform both centers and orients: every
        # objective reads larger-is-better, and inv() returns raw units
        self.ytransform = AffineTransform(scale=self.sign, shift=-self.ydata.mean())
        self.centered_y = self.ytransform(self.ydata)

    @property
    def sign(self):
        return 1.0 if self.maximize else -1.0

    def best_action(self):
        # centering does not move the argmax, and centered_y is larger-is-better
        return self.xdata[np.argmax(self.centered_y)]

    def add_points(self, actions: np.ndarray, values: np.ndarray):
        self.xdata = np.vstack([self.xdata, np.atleast_2d(actions)])
        self.ydata = np.hstack([self.ydata, values])
        self.__post_init__()

    @classmethod
    def from_df(cls, df, column, action_columns, maximize, name=None):
        return cls(
            name        = column if name is None else name,
            maximize    = maximize,
            ydata       = df[column].to_numpy(dtype=float),
            xdata       = df[action_columns].to_numpy(dtype=float),
            column      = column
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
        # one frame spanning every objective's actions, so actions taken from
        # different objectives are normalized into the same box
        if not self.objectives:
            self.xtransform = None
            return

        self.xtransform = AffineTransform.make_normalized(
            data=np.concatenate([o.xdata for o in self.objectives]), axis=0
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
        """Centered values of the selected objectives, each larger-is-better."""
        return self._unwrap([o.centered_y for o in self._select(objs)], objs)

    def actions(self, objs=None):
        """Actions of the selected objectives, in the shared normalized frame."""
        return self._unwrap([self.xtransform(o.xdata) for o in self._select(objs)], objs)

    def max(self, objs=None):
        """Largest centered value of each selected objective."""
        return self._unwrap(
            np.array([o.centered_y.max() for o in self._select(objs)]), objs
        )

    def min(self, objs=None):
        """Smallest centered value of each selected objective."""
        return self._unwrap(
            np.array([o.centered_y.min() for o in self._select(objs)]), objs
        )

    def range(self, objs=None):
        """Per-objective [min, max] of the centered values. Shape (num_objs, 2),
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