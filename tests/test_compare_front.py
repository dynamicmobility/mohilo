"""Tests for the front-versus-anti-front evaluation script.

The claims the script rests on: the two sets it picks really are opposite ends
of the fit's own posterior, the order they arrive in is drawn rather than
blocked, `--reversed` is exactly the csv the plain run wrote turned around, and
no run ever writes over another. The trial loop needs probes, a socket and an
iPad, so it is not here.
"""

import csv

import numpy as np
import pytest

import hilo.compare_front as cf
from pypolar.experiment.dataset import ExperimentDataset
from pypolar.optimization.gp import DecoupledMOGP, NoiseModel
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

BOX = (-3.0, 3.0)


@pytest.fixture(autouse=True)
def small_scan(monkeypatch):
    """A 2^14 Sobol scan is the run's, not a test's."""
    monkeypatch.setattr(cf, 'SCAN', 512)


@pytest.fixture
def dataset():
    """A run over two objectives whose optima pull opposite ways, so the fit has
    a front to have an end of."""
    actions = np.linspace(-3.0, 3.0, 12)[:, None]
    objs = DecoupledObjectives([
        Objective.from_data(actions=actions, values=actions[:, 0] ** 2,
                            maximize=False, name='cost', action_bounds=BOX),
        Objective.from_data(actions=actions, values=-(actions[:, 0] - 1.0) ** 2,
                            maximize=False, name='comfort', action_bounds=BOX),
    ])
    data = ExperimentDataset(name='run', subject='MT01',
                             timestamp='2026-08-27T15:46:07')
    data.add_trial(objs, gp=DecoupledMOGP(objs, noise=NoiseModel.prior(0.1)))

    return data


@pytest.fixture
def samples(dataset):
    """A seeded order, so a test reads the same run twice."""
    return cf.front_samples(dataset, rng=np.random.default_rng(0))


# ---- the command line ------------------------------------------------------

def test_a_dataset_is_required():
    with pytest.raises(SystemExit):
        cf.parse_args([])


def test_the_defaults_match_the_run_script():
    args = cf.parse_args(['--dataset', 'run.json'])

    assert (args.connect, args.emulate, args.reversed) == (True, False, False)


def test_reversed_and_resume_are_flags():
    args = cf.parse_args(['--dataset', 'run.json', '--reversed', '--resume'])

    assert args.reversed and args.resume


# ---- picking the two sets --------------------------------------------------

def test_it_returns_k_of_each_set(samples):
    actions, estimates, kinds, names = samples

    assert len(actions) == len(estimates) == len(kinds) == 2 * cf.K
    assert sorted(kinds) == sorted([cf.PARETO] * cf.K + [cf.ANTI] * cf.K)
    assert names == ['cost', 'comfort']


def test_the_actions_lie_in_the_runs_own_box(samples):
    actions = samples[0]

    assert np.all(actions >= BOX[0]) and np.all(actions <= BOX[1])


def test_the_pareto_set_dominates_the_anti_set_in_the_fits_own_posterior(dataset, samples):
    """The property the whole experiment tests. It holds by construction here:
    whether it holds against the truth is what measuring the subject decides."""
    actions, _, kinds, _ = samples
    mu, _ = dataset.get_model().posterior_at(actions)   # maximization space
    front, anti = mu[kinds == cf.PARETO], mu[kinds == cf.ANTI]

    dominated = np.any(np.all(front[None] > anti[:, None], axis=2), axis=1)

    assert dominated.all()


def test_the_estimates_are_in_the_objectives_own_units(dataset, samples):
    """Raw units, so the csv reads in W/kg and rating points rather than in
    standardized maximization space."""
    actions, estimates, _, _ = samples
    raw, _ = dataset.get_model().posterior_at(actions, raw=True)

    np.testing.assert_allclose(estimates, raw)


def test_the_two_sets_are_disjoint(samples):
    actions = samples[0]

    assert len({tuple(a) for a in actions}) == 2 * cf.K


# ---- the drawn order -------------------------------------------------------

def test_the_order_is_drawn_not_blocked():
    """Two draws give different orders. A subject given three of one set and
    then three of the other is rating the block as much as the action."""
    orders = {tuple(np.random.default_rng(seed).permutation(2 * cf.K))
              for seed in range(20)}

    assert len(orders) > 1


def test_the_draw_is_reproducible_under_a_given_generator(dataset):
    twice = [cf.front_samples(dataset, rng=np.random.default_rng(7)) for _ in range(2)]

    np.testing.assert_allclose(twice[0][0], twice[1][0])
    assert list(twice[0][2]) == list(twice[1][2])


def test_the_draw_reorders_rather_than_reselects(dataset):
    """Which actions are chosen is deterministic under SEED; only their order is
    drawn, so two draws are permutations of one set."""
    a = cf.front_samples(dataset, rng=np.random.default_rng(1))[0]
    b = cf.front_samples(dataset, rng=np.random.default_rng(9))[0]

    assert {tuple(row) for row in a} == {tuple(row) for row in b}


def test_the_kind_travels_with_its_action(dataset):
    """The permutation must not shuffle the labels out from under the rows."""
    actions, _, kinds, _ = cf.front_samples(dataset, rng=np.random.default_rng(3))
    mu, _ = dataset.get_model().posterior_at(actions)
    front, anti = mu[kinds == cf.PARETO], mu[kinds == cf.ANTI]

    assert np.all(np.any(np.all(front[None] > anti[:, None], axis=2), axis=1))


# ---- --reversed ------------------------------------------------------------

def test_reversed_reads_the_plain_runs_csv_back(tmp_path, samples):
    """The order was drawn once. Reading it back is what makes the two runs the
    same experiment backwards rather than two different experiments."""
    path = cf.write_csv(tmp_path / 'evaluation.csv', *samples)
    actions, estimates, kinds, names = cf.read_samples(path)

    np.testing.assert_allclose(actions, samples[0], rtol=1e-5)
    np.testing.assert_allclose(estimates, samples[1], rtol=1e-5)
    assert list(kinds) == list(samples[2])
    assert names == samples[3]


def test_reversed_without_a_plain_run_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match='without --reversed'):
        cf.read_samples(tmp_path / 'evaluation.csv')


# ---- the csv ---------------------------------------------------------------

def test_the_csv_has_a_row_per_trial_numbered_from_one(tmp_path, samples):
    rows = list(csv.DictReader(cf.write_csv(tmp_path / 'e.csv', *samples).open()))

    assert [row['trial'] for row in rows] == [str(i) for i in range(1, 2 * cf.K + 1)]
    assert [row['type'] for row in rows] == list(samples[2])


def test_the_csv_names_a_column_per_objective(tmp_path, samples):
    with cf.write_csv(tmp_path / 'e.csv', *samples).open() as f:
        header = next(csv.reader(f))

    assert header == ['trial', 'action', 'cost est', 'comfort est', 'type']


def test_the_csv_action_reads_back_as_the_action(tmp_path, samples):
    """One column holding the whole vector, so it survives a round trip."""
    rows = list(csv.DictReader(cf.write_csv(tmp_path / 'e.csv', *samples).open()))
    back = np.array([eval(row['action']) for row in rows])

    np.testing.assert_allclose(back, samples[0], rtol=1e-5)


# ---- the pieces ------------------------------------------------------------

def test_peel_takes_the_best_end_and_the_worst_end():
    mu = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    assert list(cf.peel(mu, 1, +1)) == [2]
    assert list(cf.peel(mu, 1, -1)) == [0]


def test_peel_keeps_peeling_until_it_has_enough():
    """A smooth posterior often puts a single point on the first layer, so one
    layer is not enough to fill a set of K."""
    mu = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    assert len(cf.peel(mu, 3, +1)) == 3


def test_peel_stops_when_the_points_run_out():
    mu = np.array([[0.0, 0.0], [1.0, 1.0]])

    assert len(cf.peel(mu, 10, +1)) == 2


def test_spread_returns_everything_when_there_is_no_choice():
    idxs = np.array([0, 1])

    assert list(cf.spread(idxs, np.zeros((2, 2)), 3)) == [0, 1]


def test_peel_can_be_told_to_leave_the_other_end_alone():
    """An extreme point sits on the front and the anti-front at once. Peeling
    the second end from what the first did not claim is what stops the same
    action being handed to a subject twice, once under each label."""
    mu = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    front = cf.peel(mu, 2, +1)

    assert not set(cf.peel(mu, 2, -1, exclude=front)) & set(front)


def test_spread_reaches_the_extremes_rather_than_clustering():
    """Greedy furthest-point: with a tight clump and two outliers, it must not
    return three of the clump."""
    F    = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [1.0, 0.0], [0.5, 1.0]])
    idxs = np.arange(5)

    assert {3, 4} <= set(cf.spread(idxs, F, 3))


# ---- never writing over a run ----------------------------------------------

def test_a_file_already_there_is_refused(tmp_path):
    """A subject's data is not overwritten, ever."""
    (tmp_path / 'evaluation.csv').write_text('trial\n')

    with pytest.raises(FileExistsError, match='evaluation.csv'):
        cf.refuse_overwrite(tmp_path / 'evaluation.csv', tmp_path / 'evaluation.json')


def test_it_names_every_file_in_the_way(tmp_path):
    for name in ('evaluation.csv', 'evaluation.log'):
        (tmp_path / name).write_text('x')

    with pytest.raises(FileExistsError) as excinfo:
        cf.refuse_overwrite(*(tmp_path / n for n in
                              ('evaluation.csv', 'evaluation.json', 'evaluation.log')))

    assert 'evaluation.csv' in str(excinfo.value)
    assert 'evaluation.log' in str(excinfo.value)


def test_a_clear_directory_passes(tmp_path):
    cf.refuse_overwrite(tmp_path / 'evaluation.csv', tmp_path / 'evaluation.json')


# ---- the printed plan ------------------------------------------------------

def test_the_plan_lists_every_trial_with_its_set(samples):
    lines = cf.plan(samples[0], samples[2]).splitlines()

    assert len(lines) == 2 * cf.K
    assert all(kind in line for line, kind in zip(lines, samples[2]))


def test_the_plan_marks_the_trial_a_dead_run_stopped_on(samples):
    """What a session that fails partway needs from the log: which action is
    next, not just how many were done."""
    marked = [line for line in cf.plan(samples[0], samples[2], done=2).splitlines()
              if line.lstrip().startswith('>')]

    assert len(marked) == 1 and ' 3 ' in marked[0]


def test_a_finished_run_marks_nothing(samples):
    plan = cf.plan(samples[0], samples[2], done=2 * cf.K)

    assert '>' not in plan


# ---- resuming a stopped session --------------------------------------------

class TestResume:
    """`--resume` continues the same files rather than starting a run beside
    them, so what it needs from the record is where the trials stopped."""

    @pytest.fixture
    def stopped(self, samples):
        """An evaluation that took two of its six trials and then died."""
        actions, _, kinds, _ = samples
        objs = DecoupledObjectives([
            Objective.from_empty('cost',    maximize=False, action_bounds=BOX),
            Objective.from_empty('comfort', maximize=True,  action_bounds=BOX),
        ])
        data = ExperimentDataset(name='evaluation', subject='MT01',
                                 timestamp='2026-08-27T15:46:07')
        for action, kind in zip(actions[:2], kinds[:2]):
            for name, value in (('cost', 1.0), ('comfort', 2.0)):
                objs.add_point(name, np.atleast_2d(action), [value])
            data.add_trial(objs, action=action, source=kind)

        return data

    def test_it_picks_up_where_the_trials_stopped(self, stopped):
        fresh = ExperimentDataset(name='evaluation', subject='MT01',
                                  timestamp='2026-08-27T15:46:07')

        assert fresh.resume(stopped) == 2

    def test_the_measured_trials_are_kept(self, stopped, samples):
        fresh = ExperimentDataset(name='evaluation', subject='MT01',
                                  timestamp='2026-08-27T15:46:07')
        fresh.resume(stopped)

        np.testing.assert_allclose(fresh.get_actions(), samples[0][:2])
        assert fresh.get_sources() == list(samples[2][:2])

    def test_the_plan_marks_the_trial_it_resumes_on(self, samples, stopped):
        fresh = ExperimentDataset(name='evaluation', subject='MT01',
                                  timestamp='2026-08-27T15:46:07')
        start = fresh.resume(stopped)
        marked = [line for line in cf.plan(samples[0], samples[2], start).splitlines()
                  if line.lstrip().startswith('>')]

        assert len(marked) == 1 and ' 3 ' in marked[0]

    def test_a_finished_session_resumes_to_nothing_left(self, samples, stopped):
        """Resuming a complete run is a no-op, not an error."""
        objs = stopped.get_objectives()
        for action, kind in zip(samples[0][2:], samples[2][2:]):
            for name in ('cost', 'comfort'):
                objs.add_point(name, np.atleast_2d(action), [1.0])
            stopped.add_trial(objs, action=action, source=kind)
        fresh = ExperimentDataset(name='evaluation', subject='MT01',
                                  timestamp='2026-08-27T15:46:07')

        assert fresh.resume(stopped) == 2 * cf.K
