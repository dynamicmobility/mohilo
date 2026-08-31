"""Tests for the record a finished run leaves behind.

The promise a dataset makes is that everything in it is *rebuildable*: a GP
comes back from its state dict without being refit, an objective from its raw
measurements, and a groundtruth or an acquisition from the arguments that made
it. The tests are written against that -- what goes in comes back, through a
json file rather than through memory.
"""

import json

import numpy as np
import pytest

from botorch.acquisition import UpperConfidenceBound

from pypolar.experiment.dataset import ExperimentDataset, TrialDataset
from pypolar.feedback.acquisition import AcquisitionParams
from pypolar.feedback.synthetic import SyntheticOracleParams
from pypolar.optimization.gp import BoTorchGP, NoiseModel
from pypolar.optimization.objectives import DecoupledObjectives, Objective


# ---- fixtures --------------------------------------------------------------

BOX     = (-5.0, 5.0)
ACTIONS = np.linspace(-4.0, 4.0, 9)[:, None]

ACQUISITION = AcquisitionParams(strategy='ucb', seed=3, ucb_beta=2.0)
GROUNDTRUTH = SyntheticOracleParams(func='Levy', objectives=('cost',), dim=1,
                                    box=5.0, seed=3, rel_noise_std=0.1)
AUX         = {'recommended': np.array([1.5]), 'regret': 0.25}


def acqf_bounds_hold(action):
    return np.all(action >= BOX[0] - 1e-9) and np.all(action <= BOX[1] + 1e-9)


def _objectives():
    """Two objectives over the same nine actions, one measured, one empty."""
    return DecoupledObjectives([
        Objective.from_data(actions=ACTIONS, values=np.sin(ACTIONS[:, 0]),
                            maximize=False, name='cost', action_bounds=BOX),
        Objective.from_empty('comfort', maximize=True, action_bounds=BOX),
    ])


@pytest.fixture
def objectives():
    return _objectives()


@pytest.fixture
def gp(objectives):
    return BoTorchGP(objectives['cost'], noise=NoiseModel.prior(0.3),
                     min_length_scale=0.1)


@pytest.fixture
def dataset(objectives, gp):
    """A two-step run: a random opening, then one acquired action."""
    data = ExperimentDataset(name='ucb', acquisition=ACQUISITION,
                             groundtruth=GROUNDTRUTH, config={'seed': 3})
    data.add_trial(_objectives(), action=np.array([1.0]), source='random')
    data.add_trial(objectives, gp=gp, action=np.array([2.0]), source='ucb',
                   aux=AUX)

    return data


@pytest.fixture
def saved(dataset, tmp_path):
    """The same run, written out and read back."""
    return ExperimentDataset.load(dataset.save(tmp_path / 'run.json'))


# ---- recording -------------------------------------------------------------

class TestAddTrial:

    def test_trials_are_numbered_in_order(self, dataset):
        assert [trial.trial for trial in dataset] == [0, 1]
        assert len(dataset) == 2

    def test_a_step_without_a_gp_records_none_of_it(self, dataset):
        assert dataset[0].state_dict is None
        assert dataset[0].gp is None

    def test_the_gp_record_is_read_off_the_gp(self, dataset, gp):
        assert dataset[1].gp['objective'] == 'cost'
        assert dataset[1].gp['min_length_scale'] == gp.min_length_scale
        assert dataset[1].gp['noise'] == {'std': None, 'median': 0.3, 'sigma': 1.0}

    def test_every_objective_is_recorded_measured_or_not(self, dataset):
        assert set(dataset[1].measurements) == {'cost', 'comfort'}
        assert dataset[1].measurements['comfort']['ydata'].size == 0

    def test_aux_is_whatever_the_run_scored_with(self, dataset):
        # the run picks its own keys: nothing here is a field on the record
        assert dataset[0].aux == {}
        assert set(dataset[1].aux) == {'recommended', 'regret'}
        assert dataset[1].aux['regret'] == 0.25

    def test_a_scalar_aux_is_stored_as_an_array(self, dataset):
        assert isinstance(dataset[1].aux['regret'], np.ndarray)
        assert dataset[1].aux['regret'].shape == ()


# ---- the file --------------------------------------------------------------

class TestSaveAndLoad:

    def test_saving_needs_somewhere_to_save_to(self, dataset):
        with pytest.raises(ValueError):
            dataset.save()

    def test_the_file_is_json_and_carries_the_configs_fingerprint(self, dataset, tmp_path):
        payload = json.loads(dataset.save(tmp_path / 'run.json').read_text())

        assert payload['name'] == 'ucb'
        assert payload['config'] == {'seed': 3}
        assert len(payload['fingerprint']) == 64      # sha256, hex

    def test_a_second_save_needs_no_path(self, dataset, tmp_path):
        first = dataset.save(tmp_path / 'run.json')
        assert dataset.save() == first

    def test_who_and_when_are_written_and_come_back(self, objectives, tmp_path):
        data = ExperimentDataset(name='ucb', subject='MT01',
                                 timestamp='2026-08-27T15:46:07')
        back = ExperimentDataset.load(data.save(tmp_path / 'run.json'))

        assert (back.subject, back.timestamp) == ('MT01', '2026-08-27T15:46:07')

    def test_a_run_that_recorded_neither_writes_them_as_null(self, dataset, tmp_path):
        payload = json.loads(dataset.save(tmp_path / 'run.json').read_text())

        assert payload['subject'] is None and payload['timestamp'] is None

    def test_a_file_written_before_the_fields_existed_still_loads(self, dataset, tmp_path):
        """Backwards compatibility: the keys are read with `get`, so a run saved
        by an older version comes back with them as None rather than raising."""
        path    = dataset.save(tmp_path / 'run.json')
        payload = json.loads(path.read_text())
        path.write_text(json.dumps({key: value for key, value in payload.items()
                                    if key not in ('subject', 'timestamp')}))

        older = ExperimentDataset.load(path)

        assert (older.subject, older.timestamp) == (None, None)
        assert len(older) == 2                        # the rest is unaffected

    def test_the_timestamp_is_the_runs_own_not_the_files(self, tmp_path):
        """It records when the run began, so re-saving does not move it."""
        data = ExperimentDataset(name='ucb', timestamp='2026-08-27T15:46:07')
        data.save(tmp_path / 'run.json')
        data.save(tmp_path / 'again.json')

        assert ExperimentDataset.load(tmp_path / 'again.json').timestamp \
            == '2026-08-27T15:46:07'

    def test_the_trials_come_back(self, saved, dataset):
        assert len(saved) == len(dataset)
        assert saved.get_sources() == ['random', 'ucb']
        np.testing.assert_allclose(saved.get_actions(), [[1.0], [2.0]])

    def test_arrays_come_back_as_arrays(self, saved):
        assert isinstance(saved[1].action, np.ndarray)
        assert isinstance(saved[1].measurements['cost']['xdata'], np.ndarray)
        assert isinstance(saved[1].aux['recommended'], np.ndarray)

    def test_the_params_come_back_typed(self, saved):
        assert saved.groundtruth == GROUNDTRUTH
        assert saved.acquisition == ACQUISITION

    def test_a_scalar_aux_stacks_to_one_per_trial(self, saved):
        regret = saved.get_aux('regret')

        assert regret.shape == (2,)
        assert np.isnan(regret[0])         # no trial recorded it, so no value
        assert regret[1] == 0.25

    def test_a_vector_aux_keeps_its_width(self, saved):
        recommended = saved.get_aux('recommended')

        assert recommended.shape == (2, 1)
        assert np.isnan(recommended[0, 0])
        assert recommended[1, 0] == 1.5

    def test_an_aux_no_trial_recorded_is_an_error_not_a_column_of_nan(self, saved):
        with pytest.raises(ValueError):
            saved.get_aux('hypervolume_regret')

    def test_the_keys_any_trial_used_are_reported(self, saved):
        assert saved.aux_names() == {'recommended', 'regret'}


# ---- what comes back out ---------------------------------------------------

class TestGetModel:
    """The state dict is the fit, so a restored model is not refit: it is the
    same posterior the run actually queried, not a rerun of the fit that
    produced it."""

    def test_the_posterior_is_the_one_that_was_recorded(self, saved, gp):
        probe = np.linspace(-5.0, 5.0, 32)[:, None]
        mu, std = gp.posterior_at(probe, raw=True)
        restored_mu, restored_std = saved.get_model(-1).posterior_at(probe, raw=True)

        np.testing.assert_allclose(restored_mu, mu)
        np.testing.assert_allclose(restored_std, std)

    def test_the_hyperparameters_survive_the_file(self, saved, gp):
        restored = saved.get_model(-1).get_fitted_hyperparameters()
        fitted   = gp.get_fitted_hyperparameters()

        np.testing.assert_allclose(restored.lengthscale, fitted.lengthscale)
        assert restored.noise_var == fitted.noise_var
        assert restored.signal_var == fitted.signal_var

    def test_there_is_no_model_before_the_first_fit(self, saved):
        assert saved.get_model(0) is None


class TestGetObjective:

    def test_it_defaults_to_the_one_the_gp_was_fit_to(self, saved, objectives):
        objective = saved.get_objective(-1)

        assert objective.name == 'cost'
        np.testing.assert_allclose(objective.ydata, objectives['cost'].ydata)

    def test_the_transforms_are_rederived_not_stored(self, saved, objectives):
        objective = saved.get_objective(-1)

        np.testing.assert_allclose(objective.standard_y, objectives['cost'].standard_y)
        np.testing.assert_allclose(objective.normalized_x, objectives['cost'].normalized_x)

    def test_a_named_objective_is_read_instead(self, saved):
        assert saved.get_objective(-1, name='comfort').name == 'comfort'

    def test_all_of_them_come_back_together(self, saved):
        assert saved.get_objectives(-1).names == ['cost', 'comfort']


class TestGetGroundtruthAndAcquisition:

    def test_the_groundtruth_is_the_one_the_arguments_describe(self, saved):
        truth = saved.get_groundtruth()
        probe = np.array([[1.0], [2.0]])

        # the same arguments must give the same function, values and spread
        np.testing.assert_allclose(truth(probe, noise=False),
                                   GROUNDTRUTH.build()(probe, noise=False))
        assert truth.measure_spread == GROUNDTRUTH.build().measure_spread

    def test_the_reference_point_survives_the_file(self, objectives, tmp_path):
        # a hypervolume is comparable only under one reference, so the one a run
        # optimized against has to come back with the truth rather than being
        # replaced by the function's own default
        ref  = (1.5, 2.5)
        data = ExperimentDataset(
            name        = 'mo',
            groundtruth = SyntheticOracleParams(
                func = 'DTLZ2', objectives=('cost', 'comfort'), dim=3, box=None,
                num_objectives=2, ref_point=ref),
            path        = tmp_path / 'mo.json'
        )
        data.add_trial(objectives)
        back = ExperimentDataset.load(data.save())

        assert back.groundtruth.ref_point == ref
        np.testing.assert_allclose(back.get_groundtruth().ref_point, ref)

    def test_without_one_the_truth_falls_back_to_its_own(self, objectives, tmp_path):
        params = SyntheticOracleParams(func='DTLZ2', objectives=('cost', 'comfort'),
                                       dim=3, box=None, num_objectives=2)
        data   = ExperimentDataset(name='mo', groundtruth=params,
                                   path=tmp_path / 'mo.json')
        data.add_trial(objectives)
        back = ExperimentDataset.load(data.save())

        assert back.groundtruth.ref_point is None
        np.testing.assert_allclose(back.get_groundtruth().ref_point, [1.1, 1.1])

    def test_a_run_with_no_groundtruth_says_so(self, objectives):
        # which is every real study: nothing knows the truth to record
        data = ExperimentDataset(name='study', acquisition=ACQUISITION)
        data.add_trial(objectives)

        with pytest.raises(ValueError):
            data.get_groundtruth()

    def test_a_run_with_no_acquisition_says_so(self, objectives):
        data = ExperimentDataset(name='random-only')
        data.add_trial(objectives)

        with pytest.raises(ValueError):
            data.get_acquisition()

    def test_the_acquisition_rebuilds_as_the_recorded_strategy(self, saved):
        acqf = saved.get_acquisition()

        assert acqf.acqf.func is UpperConfidenceBound
        assert acqf.acqf.keywords['beta'] == ACQUISITION.ucb_beta

    def test_the_rebuilt_acquisition_searches_that_trials_box(self, saved):
        # it carries no box of its own; the model it is queried against
        # supplies the objective's own action_bounds
        action = saved.get_acquisition().query(saved.get_model(-1))

        assert acqf_bounds_hold(action)


# ---- the summary -----------------------------------------------------------

class TestStr:

    def test_it_reports_who_when_and_how_far(self, dataset):
        dataset.subject, dataset.timestamp = 'MT01', '2026-08-27T15:46:07'
        report = str(dataset)

        assert report.startswith('MT01 from')
        assert 'run started     : 2026-08-27T15:46:07' in report
        assert 'trials recorded : 2' in report
        assert 'actions applied : 2' in report
        assert "{'random': 1, 'ucb': 1}" in report

    def test_it_falls_back_to_the_name_when_no_subject_was_recorded(self, dataset):
        dataset.timestamp = '2026-08-27T15:46:07'
        assert str(dataset).startswith('ucb from')

    def test_it_reports_each_objectives_measurements(self, dataset):
        dataset.timestamp = '2026-08-27T15:46:07'
        report = str(dataset)

        assert 'cost' in report and 'N=9' in report
        assert 'comfort' in report and 'empty' in report

    def test_an_empty_run_still_summarizes(self):
        report = str(ExperimentDataset(name='ucb', timestamp='2026-08-27T15:46:07'))

        assert 'trials recorded : 0' in report
        assert 'chosen by       : nothing' in report

    def test_a_run_with_no_timestamp_refuses_to_summarize(self, dataset):
        """A directory name is not a record of when a run was taken, so there is
        no fallback: the field is the only source."""
        with pytest.raises(ValueError, match='timestamp'):
            str(dataset)

    def test_repr_still_works_without_a_timestamp(self, dataset):
        """__str__ raising must not make the object undebuggable."""
        assert 'ExperimentDataset' in repr(dataset)


# ---- carrying a run on -----------------------------------------------------

class TestResume:

    @pytest.fixture
    def prior(self, dataset):
        dataset.subject, dataset.timestamp = 'MT01', '2026-08-27T15:46:07'
        return dataset

    @pytest.fixture
    def fresh(self):
        return ExperimentDataset(name='ucb', subject='MT01',
                                 timestamp='2026-08-28T09:00:00', config={'seed': 3})

    def test_the_prior_trials_are_copied_across(self, fresh, prior):
        assert fresh.resume(prior) == 2
        assert len(fresh) == 2
        np.testing.assert_allclose(fresh.get_actions(), [[1.0], [2.0]])

    def test_the_closing_record_of_a_finished_run_is_dropped(self, fresh, prior, objectives):
        """It carries the fit to everything and no action; the resumed run
        appends its own, so keeping it would leave two."""
        prior.add_trial(objectives)

        assert fresh.resume(prior) == 2
        assert len(fresh) == 2
        assert all(trial.action is not None for trial in fresh.trials)

    def test_a_new_trial_numbers_on_from_the_copied_ones(self, fresh, prior, objectives):
        fresh.resume(prior)

        assert fresh.add_trial(objectives, action=np.array([3.0])).trial == 2

    def test_the_prior_run_is_not_written_to(self, fresh, prior, objectives, tmp_path):
        """The resumed run writes its own file, so the original is untouched."""
        path = prior.save(tmp_path / 'prior.json')
        fresh.resume(prior)
        fresh.add_trial(objectives, action=np.array([3.0]))
        fresh.save(tmp_path / 'resumed.json')

        assert len(ExperimentDataset.load(path)) == 2

    def test_a_run_with_no_timestamp_cannot_be_carried_on(self, fresh, dataset):
        with pytest.raises(ValueError, match='timestamp'):
            fresh.resume(dataset)

    def test_a_run_on_another_subject_is_refused(self, fresh, prior):
        prior.subject = 'MT07'

        with pytest.raises(ValueError, match='MT07'):
            fresh.resume(prior)

    def test_a_subject_neither_run_recorded_is_not_a_mismatch(self, prior):
        """An older file records no subject, so there is nothing to compare."""
        prior.subject = None
        fresh = ExperimentDataset(name='ucb', timestamp='2026-08-28T09:00:00',
                                  config={'seed': 3})

        assert fresh.resume(prior) == 2

    def test_a_changed_configuration_warns_but_carries_on(self, prior, capsys):
        """The constants a run was made under may legitimately change between
        sessions; refusing would make it unresumable for it."""
        fresh = ExperimentDataset(name='ucb', subject='MT01',
                                  timestamp='2026-08-28T09:00:00', config={'seed': 4})

        assert fresh.resume(prior) == 2
        assert 'different configuration' in capsys.readouterr().out
