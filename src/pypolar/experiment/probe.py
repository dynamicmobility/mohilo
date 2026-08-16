"""What a measurement is: one action in, values out, over however long the
instrument takes."""

import time
from abc import ABC, abstractmethod

import numpy as np


class Probe(ABC):
    """One measurement of one or more objectives at one action.

    This is the interface a study and a simulation share. Both are handed an
    action and both hand back values; only the duration and the plumbing
    differ. A loop written against a `Probe` therefore runs unchanged against
    synthetic groundtruth, which is what lets a whole session be rehearsed
    before a subject arrives.

    Attributes:
        names: the objectives this probe reports.
        duration: nominal seconds one measurement takes.
    """

    names   : tuple
    duration: float = 0.0
    repeats : int   = 1

    @abstractmethod
    def measure(self, action, record=None):
        """Values at one action, as name -> (k,) array.

        An objective may report any number of values per measurement. One
        metabolic cost and four comfort ratings from the same trial is the
        decoupled case, and both sets of values belong to this one action.

        Args:
            action: (d,) action, in raw units.
            record: called as `record(kind, **fields)` for each observation as
                it arrives. A probe that streams through this loses nothing
                when the process dies partway through the measurement.

        Returns:
            dict of objective name -> (k,) values.
        """


class SyntheticProbe(Probe):
    """Callables standing in for instruments.

    Args:
        functions: objective name -> callable taking (n, d) actions and
            returning (n,) values. `SyntheticFunction` is the intended one.
        repeats: values drawn per measurement, an int for every objective or a
            dict keyed by name. A probe reporting four comfort ratings a trial
            sets four for that objective. Each repeat is a separate call, so
            each carries its own noise draw.
        duration: seconds one measurement takes, spread evenly over the values
            it reports. 0.0 returns immediately; a real duration makes a dress
            rehearsal take as long as the session it rehearses, and makes an
            interruption land partway through a trial as it would in the lab.
    """

    def __init__(self, functions: dict, repeats=1, duration=0.0):
        self.functions = dict(functions)
        self.duration  = float(duration)
        self.names     = tuple(self.functions)
        self.repeats   = ({name: int(repeats.get(name, 1)) for name in self.names}
                          if isinstance(repeats, dict) else
                          {name: int(repeats) for name in self.names})

        if any(k < 1 for k in self.repeats.values()):
            raise ValueError(f'repeats must be at least 1, got {self.repeats}')

    def measure(self, action, record=None):
        action = np.atleast_2d(np.asarray(action, dtype=float))
        if len(action) != 1:
            raise ValueError(f'a probe measures one action, got {len(action)}')

        values = {name: np.concatenate([np.atleast_1d(f(action))
                                        for _ in range(self.repeats[name])])
                  for name, f in self.functions.items()}

        pause = self.duration / sum(self.repeats.values())
        for name, drawn in values.items():
            for value in drawn:
                if pause:
                    time.sleep(pause)
                if record is not None:
                    record('sample', objective=name, value=float(value))

        return values
