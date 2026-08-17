"""iPad comfort survey: a 1 - 5 scale that opens one question at a time.

Run this on the external computer, then open the printed URL in Safari on the
iPad. The page is greyed out and untouchable until `ask` opens a question; from
then the subject has `timeout` seconds to pick a position and hit Submit.

    pip install websockets

Usage:

    from survey import Survey

    s = Survey()
    s.wait_for_ipad()

    s.ask()               # (1,) rating, or (0,) if the window closed unanswered
    s.close()

`ask` is what a `pypolar.Probe` calls, from the probe's own worker thread. The
empty array is how a missed question reaches the experiment: `Probe` still counts
the repeat, `Probe.data` drops it, and `Logger.end_trial` adds nothing to the
objective for a trial whose every repeat timed out.
"""

import time
from pathlib import Path

import numpy as np

from panel import HTTP_PORT, WS_PORT, Panel

PAGE    = 'survey.html'
TIMEOUT = 25.0                # seconds the subject has to answer
PERIOD  = 30.0                # wall clock one question occupies, answered early or not


class Survey(Panel):
    """The panel's rating scale, armed one question at a time.

    Args:
        timeout: seconds a question stays open.
        period: seconds the whole question occupies, so a subject who answers
            early waits out the rest before the next one opens. A period below
            the timeout just means no wait, never a shortened question.
        http_port, ws_port, quiet: as `Panel`.
    """

    def __init__(self, timeout=TIMEOUT, period=PERIOD, http_port=HTTP_PORT,
                 ws_port=WS_PORT, quiet=True):
        # set before the servers start, since a client may connect immediately
        self.timeout   = float(timeout)
        self.period    = float(period)
        self.trial     = None      # trial the caller last named
        self.repeat    = 0         # questions asked so far within it, 1-based
        self._deadline = None

        super().__init__(http_port=http_port, ws_port=ws_port, quiet=quiet,
                         directory=str(Path(__file__).resolve().parent),
                         page=PAGE)

    def ask(self, action=None, trial=None, timeout=None, period=None) -> np.ndarray:
        """Opens one question and blocks out the full period.

        Args:
            action: what the subject is rating. Unused here; the probe passes it.
            trial: which trial this question belongs to. A `Probe` calls this
                once per repeat with the arguments it was given, so a repeated
                trial is how the repeats within one action are counted.
            timeout: seconds to wait, defaulting to the survey's own.
            period: seconds the call takes in total, defaulting to the survey's.

        Returns:
            (1,) the rating, or (0,) when the window closed unanswered.
        """
        timeout = self.timeout if timeout is None else float(timeout)
        period  = self.period  if period  is None else float(period)
        start   = time.monotonic()

        if trial is not None and trial != self.trial:
            self.trial  = trial
            self.repeat = 0
        self.repeat += 1

        # a tap that landed after the last window closed is not this answer
        self.clear_sends()
        self.arm(timeout, trial=self.trial, repeat=self.repeat)
        try:
            value = self.wait_for_send(timeout=timeout, discard_stale=False)
        finally:
            self.disarm()

        self.hold(period - (time.monotonic() - start))

        return np.empty(0) if value is None else np.atleast_1d(float(value))

    def hold(self, seconds):
        """Blocks for the rest of the period, counting it down on the grey page."""
        if seconds <= 0:
            return

        self._send({'hold': seconds})
        time.sleep(seconds)

    def arm(self, seconds=None, trial=None, repeat=None):
        """Makes the scale interactable and starts the countdown.

        `trial` and `repeat` are labels for the page: a repeat above the first
        is the same action rated again, which the page says out loud.
        """
        seconds = self.timeout if seconds is None else float(seconds)
        self._deadline = time.monotonic() + seconds
        self._send({'arm': seconds, 'trial': trial, 'repeat': repeat})

    def disarm(self):
        """Greys the scale out again."""
        self._deadline = None
        self._send({'disarm': True})

    @property
    def seconds_left(self) -> float:
        """Seconds left on the open question, 0.0 when none is open."""
        if self._deadline is None:
            return 0.0

        return max(0.0, self._deadline - time.monotonic())


if __name__ == '__main__':
    s = Survey()
    print('Waiting for the iPad...')
    s.wait_for_ipad()

    # two repeats per trial, as a Probe with repeats=2 would ask them
    try:
        trial = 0
        while True:
            trial += 1
            for _ in range(2):
                print(f'Asking, trial {trial}...')
                value = s.ask(trial=trial)
                print('got', value if len(value) else 'nothing (timed out)')
    except KeyboardInterrupt:
        s.close()
