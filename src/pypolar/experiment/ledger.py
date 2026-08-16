"""The record a session leaves behind: an append-only event log, and the replay
that turns it back into objectives."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pypolar.optimization.objectives import DecoupledObjectives

SESSION         = 'session'
TRIAL_OPENED    = 'trial_opened'
TRIAL_CLOSED    = 'trial_closed'
TRIAL_ABANDONED = 'trial_abandoned'


def jsonable(value):
    """numpy scalars, arrays and paths as plain Python, so json.dumps takes them."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)

    return value


def fingerprint(config):
    """A hash of the configuration a session ran under.

    Canonical (sorted-key) JSON, so a config differing only in insertion order
    hashes the same and a config differing in any value does not. Entries json
    cannot encode fall back to `repr`, which is what makes a class -- an
    acquisition, a groundtruth -- hash by its own name.
    """
    return hashlib.sha256(
        json.dumps(jsonable(config), sort_keys=True, default=repr).encode()
    ).hexdigest()


def read_events(path):
    """Every event in a log, as a list of dicts.

    A final line that does not parse is dropped: a process killed partway
    through a write leaves one, and it is the only line that can be incomplete
    since every earlier one was flushed before the next began.
    """
    events = []
    with Path(path).open() as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise

    return events


class Ledger:
    """An append-only JSONL record of one session.

    Every event is written and flushed to disk the moment it happens, so a
    process that dies mid-trial loses at most the event in flight. Nothing is
    ever rewritten: the file is the history, and the objectives, the GP and
    everything else are *derived* from it by `replay`. That is why a crash
    costs nothing beyond the interrupted measurement -- the model holds no
    state that the log does not already contain.

    Each line is one JSON object carrying `kind`, `run`, `clock` and `t`:

        session          the configuration a run started under
        trial_opened     an action was chosen and applied
        sample           one observation inside a trial
        trial_closed     the trial's values, per objective
        trial_abandoned  the trial produced nothing usable

    `clock` is wall-clock UTC and so compares across runs; `t` is monotonic
    seconds since this run attached, immune to a clock step but meaningless
    across processes. `run` counts attaches, so which of the two applies is
    never a guess.

    Args:
        path: the log file. Its parent is created if it does not exist.
        config: anything json can encode, hashed to pin what this session is.
        resume: continue an existing log. False refuses to touch one, so a
            session cannot silently write into another's record.
    """

    def __init__(self, path, config, resume=True):
        self.path   = Path(path)
        self.config = config
        self.events = []
        self.run    = 0
        self._open  = None      # the trial in progress, if any

        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    f'{self.path} already holds a session; pass resume=True to '
                    'continue it, or write to another path'
                )
            self.events = read_events(self.path)
            self._check_config()
            self.run = max((e['run'] for e in self.events), default=-1) + 1

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('a')
        self._t0   = time.monotonic()
        self.append(SESSION, config=config, fingerprint=fingerprint(config))

    def _check_config(self):
        """A resumed run must be the run it is resuming.

        The actions in the log are numbers whose meaning is fixed by the action
        box, the objective names and their directions. Continuing under a
        changed configuration would mix two studies into one file, and nothing
        downstream could tell them apart.
        """
        started = next((e for e in self.events if e['kind'] == SESSION), None)
        if started is None or started['fingerprint'] == fingerprint(self.config):
            return

        was, now = started['config'], jsonable(self.config)
        changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
        raise ValueError(
            f'{self.path} was recorded under a different configuration; '
            f'changed: {", ".join(changed) if changed else "encoding only"}'
        )

    # ---- writing -----------------------------------------------------------

    def append(self, kind, **fields):
        """One event, on disk before this returns."""
        event = {
            'kind' : kind,
            'run'  : self.run,
            'clock': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            't'    : round(time.monotonic() - self._t0, 4),
            **jsonable(fields)
        }
        self.events.append(event)
        self._file.write(json.dumps(event, default=repr) + '\n')
        self._file.flush()
        os.fsync(self._file.fileno())   # survives a power loss, not merely a crash

        return event

    @property
    def next_trial(self):
        """The index the next trial takes, one past the highest recorded."""
        return max((e['trial'] for e in self.events if 'trial' in e), default=-1) + 1

    def open_trial(self, action, source=None, **fields):
        """Records the action a trial is about to run at, and returns its index.

        Args:
            action: (d,) action in raw units.
            source: what chose it -- an acquisition's name, 'seed', 'manual'.
        """
        if self._open is not None:
            raise RuntimeError(f'trial {self._open} is still open')

        self._open = self.next_trial
        self.append(TRIAL_OPENED, trial=self._open,
                    action=np.asarray(action, dtype=float).ravel(),
                    source=source, **fields)

        return self._open

    def record(self, kind='sample', **fields):
        """One observation inside the trial in progress.

        Bound, this is the callback a `Probe` streams through:
        `probe.measure(action, record=ledger.record)`.
        """
        if self._open is None:
            raise RuntimeError('no trial is open')

        return self.append(kind, trial=self._open, **fields)

    def close_trial(self, values, **fields):
        """Completes the open trial with its values, name -> one or more each."""
        if self._open is None:
            raise RuntimeError('no trial is open')

        event = self.append(TRIAL_CLOSED, trial=self._open,
                            values={name: np.atleast_1d(v).astype(float)
                                    for name, v in values.items()}, **fields)
        self._open = None

        return event

    def abandon_trial(self, reason, trial=None, **fields):
        """Marks a trial unusable, so replay skips it.

        `trial` defaults to the open one; naming an earlier index is how a
        resumed run closes out a trial a crash left open.
        """
        trial = self._open if trial is None else trial
        if trial is None:
            raise RuntimeError('no trial is open')

        event = self.append(TRIAL_ABANDONED, trial=trial, reason=reason, **fields)
        if trial == self._open:
            self._open = None

        return event

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---- reading -----------------------------------------------------------

    def trials(self):
        """Every completed trial, as dicts of index, action, source and values,
        in the order they were closed."""
        actions = {e['trial']: e for e in self.events if e['kind'] == TRIAL_OPENED}
        return [{'trial' : e['trial'],
                 'action': np.asarray(actions[e['trial']]['action'], dtype=float),
                 'source': actions[e['trial']].get('source'),
                 'values': {n: np.asarray(v, dtype=float)
                            for n, v in e['values'].items()}}
                for e in self.events if e['kind'] == TRIAL_CLOSED]

    def samples(self, trial):
        """Every non-bookkeeping event recorded inside one trial, in order."""
        return [e for e in self.events
                if e.get('trial') == trial
                and e['kind'] not in (TRIAL_OPENED, TRIAL_CLOSED, TRIAL_ABANDONED)]

    def replay(self, objectives):
        """Feeds every completed trial into `objectives`, in the recorded order.

        This is what resuming *is*: measurements are the only state a session
        holds, so replaying them rebuilds the objectives exactly, and the GP is
        refit from there.

        Args:
            objectives: a `DecoupledObjectives` or a single `Objective`, added
                to in place. Its names must cover every name in the log.

        Returns:
            The indices of trials opened and never closed -- a crash or an
            abort mid-measurement. Whatever they did record is still in the
            log; whether it is usable is the caller's call.
        """
        collection = (objectives if isinstance(objectives, DecoupledObjectives)
                      else DecoupledObjectives([objectives]))

        actions, opened, measured = {}, set(), {}
        for event in self.events:
            if event['kind'] == TRIAL_OPENED:
                actions[event['trial']] = np.asarray(event['action'], dtype=float)
                opened.add(event['trial'])

            elif event['kind'] == TRIAL_ABANDONED:
                opened.discard(event['trial'])

            elif event['kind'] == TRIAL_CLOSED:
                opened.discard(event['trial'])
                action = actions[event['trial']]
                for name, values in event['values'].items():
                    if name not in collection.names:
                        raise ValueError(
                            f'trial {event["trial"]} recorded {name!r}, which is '
                            f'not among the objectives {collection.names}'
                        )
                    values = np.atleast_1d(values).astype(float)
                    x, y = measured.setdefault(name, ([], []))
                    x.append(np.tile(action, (len(values), 1)))
                    y.append(values)

        # added per objective rather than per trial: the objectives are
        # decoupled, so interleaving them changes nothing, and one append
        # apiece keeps a resume off the N = 1 transforms it would pass through
        for name, (x, y) in measured.items():
            collection.add_point(name, np.vstack(x), np.concatenate(y))

        return sorted(opened)
