"""Tests for the run script's command line and the dataset a resume names.

Only the pieces that touch no hardware are exercised: argument parsing and
finding the file `--resume` points at. Carrying a run on lives on
`ExperimentDataset` and is tested with it; the trial loop needs probes, a
socket and an iPad, so it is not here either.
"""

import pytest

from hilo.fit_mogp import find_dataset, parse_args
from pypolar.experiment.dataset import ExperimentDataset


# ---- the command line ------------------------------------------------------

def test_subject_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_the_bare_command_is_a_connected_hardware_run():
    """The defaults are the ones that collect data: a forgotten flag measures a
    subject rather than quietly simulating one."""
    args = parse_args(['--subject', 'MT01'])

    assert (args.connect, args.emulate, args.resume) == (True, False, None)


@pytest.mark.parametrize('flag, field, value', [
    ('--no-connect', 'connect', False),
    ('--emulate',    'emulate', True),
    ('--no-emulate', 'emulate', False),
])
def test_each_default_is_overridable(flag, field, value):
    assert getattr(parse_args(['--subject', 'MT01', flag]), field) is value


def test_resume_is_a_path():
    args = parse_args(['--subject', 'MT01', '--resume', 'hilo/output/run.json'])

    assert args.resume.name == 'run.json'


# ---- finding the dataset to resume -----------------------------------------

def test_a_dataset_file_is_taken_as_it_is(tmp_path):
    path = ExperimentDataset(name='MT01').save(tmp_path / 'MT01.json')

    assert find_dataset(path) == path


def test_a_directory_is_refused(tmp_path):
    """A run is resumed from a dataset, not from the directory holding one."""
    ExperimentDataset(name='MT01').save(tmp_path / 'MT01.json')

    with pytest.raises(FileNotFoundError):
        find_dataset(tmp_path)


def test_a_path_that_is_not_there_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset(tmp_path / 'missing.json')
