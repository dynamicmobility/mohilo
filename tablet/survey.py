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

from panel import HTTP_PORT, WS_PORT, Panel, local_ip

PAGE    = 'survey.html'
TIMEOUT = 25.0                # seconds the subject has to answer


class Survey(Panel):
    """The panel's rating scale, armed one question at a time.

    Args:
        timeout: seconds a question stays open.
        http_port, ws_port, quiet: as `Panel`.
    """

    def __init__(self, timeout=TIMEOUT, http_port=HTTP_PORT, ws_port=WS_PORT,
                 quiet=True):
        # set before the servers start, since a client may connect immediately
        self.timeout   = float(timeout)
        self._deadline = None

        super().__init__(http_port=http_port, ws_port=ws_port, quiet=quiet,
                         directory=str(Path(__file__).resolve().parent))

        print(f'Survey page:  http://{local_ip()}:{http_port}/{PAGE}')

    def ask(self, action=None, timeout=None) -> np.ndarray:
        """Opens one question and blocks until it is answered or times out.

        Args:
            action: what the subject is rating. Unused here; the probe passes it.
            timeout: seconds to wait, defaulting to the survey's own.

        Returns:
            (1,) the rating, or (0,) when the window closed unanswered.
        """
        timeout = self.timeout if timeout is None else float(timeout)

        # a tap that landed after the last window closed is not this answer
        self.clear_sends()
        self.arm(timeout)
        try:
            value = self.wait_for_send(timeout=timeout, discard_stale=False)
        finally:
            self.disarm()

        return np.empty(0) if value is None else np.atleast_1d(float(value))

    def arm(self, seconds=None):
        """Makes the scale interactable and starts the countdown."""
        seconds = self.timeout if seconds is None else float(seconds)
        self._deadline = time.monotonic() + seconds
        self._send({'arm': seconds})

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

    try:
        while True:
            print('Asking...')
            value = s.ask()
            print('got', value if len(value) else 'nothing (timed out)')
            time.sleep(1.0)
    except KeyboardInterrupt:
        s.close()
