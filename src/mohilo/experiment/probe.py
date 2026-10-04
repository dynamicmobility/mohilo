"""What a measurement is: one action in, values out, over however long the
instrument takes."""

import threading
from collections.abc import Callable

import numpy as np


class Probe:
    """One instrument, measured in a background thread.

    Args:
        name: what this probe is called.
        caller: called once per repeat, returning that measurement's value.
        repeats: measurements taken per trial. 0 and 1 both take one; above
            that the probe reports several values at the one action, which is
            what a subject rating a condition four times produces.
        obj_name: the objective these values belong to.
    """

    def __init__(
        self,
        name              : str,
        caller            : Callable[..., float],
        repeats           : int = 0,
        obj_name          : str = None,
        separate_thread   : bool = True # TODO: make tests when separate_thread is false
    ):
        self.name               = name
        self.obj_name           = obj_name
        self.caller             = caller
        self.repeats            = repeats
        self.finished           = False
        self.separate_thread    = separate_thread

        self._values    = []
        self.thread     = None
        self.stop_event = threading.Event()

    @property
    def n_values(self) -> int:
        """Measurements one trial asks for."""
        return max(self.repeats, 1)

    def call_and_detect(self, *args, **kwargs):
        while len(self._values) < self.n_values and not self.stop_event.is_set(): # TODO fix this
            self._values.append(np.atleast_1d(self.caller(*args, **kwargs)))

        self.finished = len(self._values) == self.n_values

    def measure(self, *args, **kwargs):
        """Starts measuring. The values arrive in `data`, `finished` says when."""
        self.reset()

        if self.separate_thread:
            self.stop_event = threading.Event()
            self.thread     = threading.Thread(target=self.call_and_detect,
                                               args=args, kwargs=kwargs, daemon=True)
            self.thread.start()
        else:
            self.call_and_detect(*args, **kwargs)

    @property
    def data(self) -> np.ndarray:
        """(k,) values collected so far, k = `n_values` once finished."""
        if not self._values:
            return np.empty(0)

        return np.concatenate(self._values).astype(float)

    def end_measurement_thread(self):
        """Stops after the call in flight, and waits for the thread to exit."""
        if self.thread is None or not self.separate_thread:
            return

        self.stop_event.set()
        self.thread.join()
        self.thread = None

    def reset(self):
        if self.separate_thread:
            self.end_measurement_thread()
        self.finished = False
        self._values  = []
