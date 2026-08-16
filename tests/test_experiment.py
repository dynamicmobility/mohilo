"""Tests for the session record and the measurement protocol.

The properties each name promises: an *append-only* log never rewrites a line
and so survives being killed at any point, a *fingerprinted* one refuses to be
continued under a configuration it was not recorded under, and a *replay*
rebuilds exactly the objectives the recorded trials had built. A `Probe` is the
interface a study and a simulation share, so the synthetic one is tested for
the shapes and the streaming a hardware one must also provide.
"""

import json

import numpy as np
import pytest

from pypolar.experiment.ledger import Ledger, fingerprint, read_events
from pypolar.experiment.probe import SyntheticProbe
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

CONFIG = {'dim': 2, 'box': 5.0, 'strategy': 'lognei', 'names': ['cost', 'comfort']}


@pytest.fixture
def path(tmp_path):
    return tmp_path / 'session' / 'run.jsonl'


@pytest.fixture
def ledger(path):
    with Ledger(path, CONFIG) as ledger:
        yield ledger


def objectives():
    """Two empty objectives over the same 2D box, pointing opposite ways."""
    return DecoupledObjectives([
        Objective.from_empty('cost',    maximize=False, action_bounds=(-5.0, 5.0)),
        Objective.from_empty('comfort', maximize=True,  action_bounds=(-5.0, 5.0)),
    ])


def run_trial(ledger, action, values, source='test'):
    """One complete trial, as the loop would record it."""
    ledger.open_trial(action, source=source)
    ledger.close_trial(values)


# ---- fingerprint -----------------------------------------------------------

def test_fingerprint_ignores_key_order():
    assert fingerprint({'a': 1, 'b': 2}) == fingerprint({'b': 2, 'a': 1})


def test_fingerprint_tracks_values():
    assert fingerprint({'box': 5.0}) != fingerprint({'box': 4.0})


def test_fingerprint_takes_unencodable_entries():
    """A config holding a class -- an acquisition, a groundtruth -- still hashes,
    by that class's own repr."""
    assert fingerprint({'acqf': dict}) == fingerprint({'acqf': dict})
    assert fingerprint({'acqf': dict}) != fingerprint({'acqf': list})


def test_fingerprint_takes_numpy():
    assert fingerprint({'x': np.arange(3)}) == fingerprint({'x': [0, 1, 2]})


# ---- writing ---------------------------------------------------------------

def test_creates_parent_directory(path):
    Ledger(path, CONFIG).close()
    assert path.exists()


def test_session_event_opens_the_log(ledger, path):
    first = read_events(path)[0]
    assert first['kind'] == 'session'
    assert first['fingerprint'] == fingerprint(CONFIG)
    assert first['run'] == 0


def test_every_event_carries_both_clocks(ledger, path):
    run_trial(ledger, [1.0, 2.0], {'cost': 3.0})
    for event in read_events(path):
        assert isinstance(event['t'], float)
        assert event['clock'].endswith('+00:00')
        assert event['run'] == 0


def test_events_are_on_disk_before_append_returns(ledger, path):
    """The whole point of the log: a process dying immediately after an event
    still leaves that event behind."""
    ledger.open_trial([0.0, 0.0], source='seed')
    assert read_events(path)[-1]['kind'] == 'trial_opened'


def test_numpy_actions_and_values_encode(ledger, path):
    run_trial(ledger, np.array([1.0, 2.0]), {'cost': np.float64(3.0)})
    text = path.read_text()
    assert json.loads(text.splitlines()[-1])['values'] == {'cost': [3.0]}


def test_trials_are_numbered_from_zero(ledger):
    assert ledger.open_trial([0.0, 0.0]) == 0
    ledger.close_trial({'cost': 1.0})
    assert ledger.open_trial([1.0, 1.0]) == 1


def test_two_open_trials_raise(ledger):
    ledger.open_trial([0.0, 0.0])
    with pytest.raises(RuntimeError):
        ledger.open_trial([1.0, 1.0])


def test_recording_outside_a_trial_raises(ledger):
    with pytest.raises(RuntimeError):
        ledger.record(value=1.0)


def test_closing_outside_a_trial_raises(ledger):
    with pytest.raises(RuntimeError):
        ledger.close_trial({'cost': 1.0})


def test_samples_belong_to_the_open_trial(ledger):
    ledger.open_trial([0.0, 0.0])
    ledger.record(objective='comfort', value=3.0)
    ledger.record(objective='comfort', value=4.0)
    ledger.close_trial({'comfort': [3.0, 4.0]})

    assert [e['value'] for e in ledger.samples(0)] == [3.0, 4.0]


# ---- resume ----------------------------------------------------------------

def test_resume_continues_the_trial_numbering(path):
    with Ledger(path, CONFIG) as first:
        run_trial(first, [0.0, 0.0], {'cost': 1.0})
        run_trial(first, [1.0, 1.0], {'cost': 2.0})

    with Ledger(path, CONFIG) as second:
        assert second.next_trial == 2
        assert second.run == 1


def test_resume_reads_the_earlier_events(path):
    with Ledger(path, CONFIG) as first:
        run_trial(first, [0.0, 0.0], {'cost': 1.0})

    with Ledger(path, CONFIG) as second:
        assert [e['kind'] for e in second.events[:3]] == \
            ['session', 'trial_opened', 'trial_closed']


def test_resume_under_a_changed_config_raises(path):
    with Ledger(path, CONFIG) as first:
        run_trial(first, [0.0, 0.0], {'cost': 1.0})

    with pytest.raises(ValueError, match='box'):
        Ledger(path, {**CONFIG, 'box': 4.0})


def test_resume_under_a_reordered_config_is_allowed(path):
    Ledger(path, CONFIG).close()
    Ledger(path, dict(reversed(list(CONFIG.items())))).close()


def test_refusing_to_resume_raises(path):
    Ledger(path, CONFIG).close()
    with pytest.raises(FileExistsError):
        Ledger(path, CONFIG, resume=False)


def test_a_truncated_final_line_is_dropped(path):
    """A process killed partway through a write leaves one incomplete line, and
    only the last line can be incomplete."""
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [0.0, 0.0], {'cost': 1.0})

    with path.open('a') as f:
        f.write('{"kind": "trial_open')

    assert len(read_events(path)) == 3
    assert Ledger(path, CONFIG).next_trial == 1


def test_a_truncated_interior_line_raises(path):
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [0.0, 0.0], {'cost': 1.0})

    lines = path.read_text().splitlines()
    path.write_text('\n'.join([lines[0], '{"kind": "trun', *lines[1:]]) + '\n')
    with pytest.raises(json.JSONDecodeError):
        read_events(path)


# ---- replay ----------------------------------------------------------------

def test_replay_rebuilds_the_measurements(path):
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0, 'comfort': 3.0})
        run_trial(ledger, [3.0, 4.0], {'cost': 20.0, 'comfort': 5.0})

    objs = objectives()
    with Ledger(path, CONFIG) as ledger:
        assert ledger.replay(objs) == []

    np.testing.assert_allclose(objs['cost'].ydata, [10.0, 20.0])
    np.testing.assert_allclose(objs['cost'].xdata, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(objs['comfort'].ydata, [3.0, 5.0])


def test_replay_repeats_the_action_for_every_value(path):
    """Four comfort ratings and one metabolic cost from one trial: decoupled
    counts over the same action, which is what DecoupledObjectives is for."""
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0, 'comfort': [3.0, 4.0, 3.0, 5.0]})

    objs = objectives()
    Ledger(path, CONFIG).replay(objs)

    assert len(objs['cost'].ydata) == 1
    assert len(objs['comfort'].ydata) == 4
    np.testing.assert_allclose(objs['comfort'].xdata, np.tile([1.0, 2.0], (4, 1)))


def test_replay_reports_an_interrupted_trial(path):
    """The crash case: the trial was opened and its samples recorded, but no
    value was ever closed out."""
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0})
        ledger.open_trial([3.0, 4.0])
        ledger.record(objective='cost', value=99.0)

    objs = objectives()
    with Ledger(path, CONFIG) as ledger:
        assert ledger.replay(objs) == [1]

    np.testing.assert_allclose(objs['cost'].ydata, [10.0])


def test_an_interrupted_trials_samples_survive(path):
    with Ledger(path, CONFIG) as ledger:
        ledger.open_trial([3.0, 4.0])
        ledger.record(objective='cost', value=99.0)

    assert [e['value'] for e in Ledger(path, CONFIG).samples(0)] == [99.0]


def test_replay_skips_an_abandoned_trial(path):
    with Ledger(path, CONFIG) as ledger:
        ledger.open_trial([3.0, 4.0])
        ledger.abandon_trial('subject stopped walking')
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0})

    objs = objectives()
    with Ledger(path, CONFIG) as ledger:
        assert ledger.replay(objs) == []

    np.testing.assert_allclose(objs['cost'].ydata, [10.0])


def test_an_earlier_trial_can_be_abandoned_on_resume(path):
    with Ledger(path, CONFIG) as ledger:
        ledger.open_trial([3.0, 4.0])

    with Ledger(path, CONFIG) as ledger:
        ledger.abandon_trial('killed mid-trial', trial=0)
        assert ledger.replay(objectives()) == []


def test_replay_into_a_single_objective(path):
    with Ledger(path, {'names': ['cost']}) as ledger:
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0})

    objective = Objective.from_empty('cost', maximize=False, action_bounds=(-5.0, 5.0))
    Ledger(path, {'names': ['cost']}).replay(objective)

    np.testing.assert_allclose(objective.ydata, [10.0])
    np.testing.assert_allclose(objective.xdata, [[1.0, 2.0]])


def test_replay_of_an_unknown_objective_raises(path):
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [1.0, 2.0], {'effort': 10.0})

    with pytest.raises(ValueError, match='effort'):
        Ledger(path, CONFIG).replay(objectives())


def test_replay_is_the_same_as_never_crashing(path):
    """Resuming has to be indistinguishable from one uninterrupted run, since
    the measurements are the only state a session carries."""
    trials = [([1.0, 2.0], {'cost': 10.0}), ([3.0, 4.0], {'cost': 20.0})]

    straight = objectives()['cost']
    for action, values in trials:
        straight.add_points(np.atleast_2d(action), np.atleast_1d(values['cost']))

    for action, values in trials:
        with Ledger(path, CONFIG) as ledger:   # one process per trial
            run_trial(ledger, action, values)

    resumed = objectives()['cost']
    Ledger(path, CONFIG).replay(resumed)

    np.testing.assert_allclose(resumed.ydata, straight.ydata)
    np.testing.assert_allclose(resumed.xdata, straight.xdata)


def test_trials_reports_the_completed_ones(path):
    with Ledger(path, CONFIG) as ledger:
        run_trial(ledger, [1.0, 2.0], {'cost': 10.0}, source='seed')
        ledger.open_trial([9.0, 9.0])

    trials = Ledger(path, CONFIG).trials()
    assert len(trials) == 1
    assert trials[0]['source'] == 'seed'
    np.testing.assert_allclose(trials[0]['action'], [1.0, 2.0])
    np.testing.assert_allclose(trials[0]['values']['cost'], [10.0])


# ---- probes ----------------------------------------------------------------

def quadratic(X):
    return -np.sum(np.asarray(X, dtype=float) ** 2, axis=1)


def test_probe_reports_one_value_per_objective():
    probe = SyntheticProbe({'cost': quadratic})
    values = probe.measure([1.0, 2.0])

    assert set(values) == {'cost'}
    np.testing.assert_allclose(values['cost'], [-5.0])


def test_probe_repeats_are_separate_draws():
    """Four ratings of one condition are four measurements, not one copied."""
    rng   = np.random.default_rng(0)
    probe = SyntheticProbe({'comfort': lambda X: rng.standard_normal(len(X))},
                           repeats=4)
    values = probe.measure([1.0, 2.0])

    assert len(values['comfort']) == 4
    assert len(np.unique(values['comfort'])) == 4


def test_probe_repeats_are_per_objective():
    probe = SyntheticProbe({'cost': quadratic, 'comfort': quadratic},
                           repeats={'comfort': 4})
    values = probe.measure([1.0, 2.0])

    assert len(values['cost']) == 1
    assert len(values['comfort']) == 4


def test_probe_names_are_the_objectives_it_reports():
    assert SyntheticProbe({'cost': quadratic, 'comfort': quadratic}).names == \
        ('cost', 'comfort')


def test_probe_rejects_repeats_below_one():
    with pytest.raises(ValueError):
        SyntheticProbe({'cost': quadratic}, repeats=0)


def test_probe_measures_one_action():
    with pytest.raises(ValueError):
        SyntheticProbe({'cost': quadratic}).measure([[1.0, 2.0], [3.0, 4.0]])


def test_probe_streams_through_record():
    recorded = []
    probe    = SyntheticProbe({'cost': quadratic, 'comfort': quadratic},
                              repeats={'comfort': 2})
    probe.measure([1.0, 2.0], record=lambda kind, **f: recorded.append((kind, f)))

    assert [kind for kind, _ in recorded] == ['sample'] * 3
    assert [f['objective'] for _, f in recorded] == ['cost', 'comfort', 'comfort']


def test_a_probe_and_a_ledger_make_a_trial(path):
    """The two pieces together, as a loop would use them: the probe streams its
    observations into the log as they arrive, and the trial closes with the
    values that replay will rebuild."""
    probe = SyntheticProbe({'cost': quadratic, 'comfort': quadratic},
                           repeats={'comfort': 2})

    with Ledger(path, CONFIG) as ledger:
        action = np.array([1.0, 2.0])
        ledger.open_trial(action, source='seed')
        ledger.close_trial(probe.measure(action, record=ledger.record))

        assert len(ledger.samples(0)) == 3

    objs = objectives()
    Ledger(path, CONFIG).replay(objs)
    np.testing.assert_allclose(objs['cost'].ydata, [-5.0])
    np.testing.assert_allclose(objs['comfort'].ydata, [-5.0, -5.0])
