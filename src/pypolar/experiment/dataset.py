"""The record a finished run leaves behind: every trial's measurements, the GP
that chose from them, and what the run was configured with.

This is the counterpart of `Ledger`, not a replacement for it. A ledger is
written *during* a session, one event at a time, and replays into objectives --
measurements are the only state it holds, and the GP is refit from them. A
dataset is written *about* a run and carries the fits themselves, so a figure
can be redrawn, a posterior re-read, or an acquisition re-queried without
refitting anything.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pypolar.experiment.ledger import fingerprint, jsonable
from pypolar.feedback.acquisition import AcquisitionParams
from pypolar.feedback.synthetic import SyntheticOracleParams
from pypolar.optimization.gp import DTYPE, BoTorchGP, DecoupledMOGP, NoiseModel
from pypolar.optimization.objectives import DecoupledObjectives, Objective


def _gp_record(gp: BoTorchGP | DecoupledMOGP):
    """The arguments a GP was built with, read off the GP itself.
    """
    record = {
        'noise'            : asdict(gp.noise),
        'length_scale'     : gp.length_scale,
        'signal_var'       : gp.signal_var,
        'min_length_scale' : gp.min_length_scale,
    }
    if isinstance(gp, DecoupledMOGP):
        return record | {'objectives': gp.objectives.names}

    return record | {'objective': gp.objective.name}


def _aux_array(value):
    """One auxiliary quantity as a float array."""
    return np.asarray(value, dtype=float)


def _action_dim(trials):
    """K, the number of action dimensions any trial recorded, or 0 for a run
    that has recorded none yet.
    """
    for trial in trials:
        if trial.action is not None:
            return trial.action.size

        for record in trial.measurements.values():
            if np.asarray(record['xdata']).size:
                return np.asarray(record['xdata']).shape[1]

    return 0


def _trial_from_json(record):
    """One trial read back, with its arrays as arrays again."""
    # TODO: document this better
    if record['action'] is not None:
        record['action'] = np.asarray(record['action'], dtype=float)

    record['aux'] = {
        name: _aux_array(value)
        for name, value in record['aux'].items()
    }

    for measurement in record['measurements'].values():
        measurement['xdata'] = np.asarray(measurement['xdata'], dtype=float)
        measurement['ydata'] = np.asarray(measurement['ydata'], dtype=float)

    return TrialDataset(**record)


@dataclass
class TrialDataset:
    """One step of a run: the measurements in hand, the GP fit to them, and the
    action that GP chose next.

    Attributes:
        trial: the step's index.
        measurements: objective name -> that objective's `to_record`.
        action: (d,) action chosen here, in raw units, or None on the last record.
        source: what chose it -- an acquisition's name, or 'random'.
        state_dict: the fitted GP's tensors, as lists. None before the first fit.
        gp: the arguments that GP was built with.
        aux: name -> auxilliary variables. Store run-specific info here.
    """

    trial        : int
    measurements : dict[str, dict]
    action       : np.ndarray | None      = None
    source       : str | None             = None
    state_dict   : dict[str, list] | None = None
    gp           : dict | None            = None
    aux          : dict[str, np.ndarray]  = field(default_factory=dict)


@dataclass
class ExperimentDataset:
    """One run, saved as a single json file. Includes model, data, acquisition.

    Attributes:
        name: what the run is called.
        subject: who it was run on.
        timestamp: when the run began, as ISO 8601. 
        action_labels: one name per action dimension, in action order, or None
            when the run named none.
        acquisition: the arguments the run's acquisition was built from, or
            None when nothing acquired.
        groundtruth: the arguments the run's oracle was built from, or None
            when there is no groundtruth.
        config: the constants the run was made under, hashed on save so two
            runs that differ in configuration cannot be mistaken for one.
        trials: the run, one record per step.
        path: where it was last written, or read from.
    """

    name         : str
    subject      : str | None                    = None
    timestamp    : str | None                    = None
    action_labels: list[str] | None              = None
    acquisition  : AcquisitionParams | None      = None
    groundtruth  : SyntheticOracleParams | None  = None
    config       : dict                          = field(default_factory=dict)
    trials       : list[TrialDataset]            = field(default_factory=list)
    path         : Path | None                   = None

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, trial):
        return self.trials[trial]

    def __str__(self):
        """Pretty-print
        """
        if self.timestamp is None:
            raise ValueError(f'{self.name} recorded no timestamp; it predates '
                             'the field, and a directory name is not a record '
                             'of when a run was taken')

        sources = {source: self.get_sources().count(source)
                   for source in dict.fromkeys(self.get_sources())}
        lines = [
            f'{self.subject or self.name} from {self.path}',
            f'  run started     : {self.timestamp}',
            f'  trials recorded : {len(self)}',
            f'  actions applied : {len(self.get_actions())}',
            f'  chosen by       : {sources or "nothing"}',
            f'  acquisition     : {None if self.acquisition is None else self.acquisition.strategy}',
            f'  fingerprint     : {fingerprint(self.config)}',
        ]
        for name, record in (self.trials[-1].measurements if len(self) else {}).items():
            ydata = np.asarray(record['ydata'], dtype=float)
            span  = ('empty' if not ydata.size
                     else f'[{ydata.min():.4g}, {ydata.max():.4g}]')
            lines.append(f'  {name:<15} : N={ydata.size}, range {span}')

        return '\n'.join(lines)

    def resume(self, prior):
        """Carries a prior run's trials into this one, and reports where its
        loop stopped.

        Args:
            prior: the `ExperimentDataset` to carry on from.

        Returns:
            the number of applied actions, which is the trial the loop picks
            up at.

        Raises:
            ValueError: `prior` recorded no `timestamp`, or was taken on a
                different subject.
        """
        if prior.timestamp is None:
            raise ValueError(f'{prior.name} recorded no timestamp; it cannot be '
                             'resumed, since there is no trustworthy record of '
                             'when it was taken')

        if None not in (prior.subject, self.subject) and prior.subject != self.subject:
            raise ValueError(f'{prior.path} was taken on {prior.subject}, not '
                             f'{self.subject}')

        if fingerprint(prior.config) != fingerprint(self.config):
            print(f'WARNING in ExperimentDataset.resume: {prior.path} was run '
                  'under a different configuration; carrying it on anyway')

        trials = list(prior.trials)
        while trials and trials[-1].action is None:
            trials.pop()

        self.trials = trials

        return len(self.get_actions())

    def add_trial(self, objectives, gp=None, action=None, source=None, aux=None):
        """Records one step, and returns the `TrialDataset` it appended.

        Args:
            objectives: the `DecoupledObjectives` as they stand *before* the
                action is applied.
            gp: the `BoTorchGP` fit to them, or None when none was fit yet.
            action, source: as on `TrialDataset`.
            aux: name -> anything `np.asarray` takes, stored as float arrays.
        """
        record = TrialDataset(
            trial        = len(self.trials),
            measurements = {name: objectives[name].to_record()
                            for name in objectives.names},
            action       = None if action is None
                           else np.asarray(action, dtype=float).ravel(),
            source       = source,
            state_dict   = None if gp is None
                           else {key: value.tolist()
                                 for key, value in gp.model.state_dict().items()},
            gp           = None if gp is None else _gp_record(gp),
            aux          = {name: _aux_array(value)
                            for name, value in (aux or {}).items()},
        )
        self.trials.append(record)

        return record

    def save(self, path=None):
        """Writes the run to json, and returns where it went.

        Args:
            path: the file to write. Defaults to wherever this dataset was last
                written or read from.
        """
        if path is None and self.path is None:
            raise ValueError('no path to save to: pass one here, or set `path` '
                             'when the dataset is made')

        self.path = Path(path or self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'name'          : self.name,
            'subject'       : self.subject,
            'timestamp'     : self.timestamp,
            'action_labels' : self.action_labels,
            'fingerprint'   : fingerprint(self.config),
            'config'        : self.config,
            'acquisition'   : None if self.acquisition is None
                              else asdict(self.acquisition),
            'groundtruth'   : None if self.groundtruth is None
                              else asdict(self.groundtruth),
            'trials'        : [asdict(trial) for trial in self.trials],
        }
        # a state dict's constraint buffers are infinite, which python's json
        # writes as `Infinity` and reads back; no stricter reader is promised
        self.path.write_text(json.dumps(jsonable(payload), indent=2))

        return self.path

    @classmethod
    def load(cls, path: Path):
        """A run read back from `save`."""
        path        = Path(path)
        payload     = json.loads(path.read_text())
        acquisition = payload['acquisition']
        groundtruth = payload['groundtruth']
        dataset     = cls(
            name          = payload['name'],
            subject       = payload.get('subject'),
            timestamp     = payload.get('timestamp'),
            action_labels = payload.get('action_labels'),
            acquisition   = None if acquisition is None
                            else AcquisitionParams(**acquisition),
            groundtruth   = None if groundtruth is None
                            else SyntheticOracleParams(**groundtruth),
            config        = payload['config'],
            path          = path,
        )
        dataset.trials = [_trial_from_json(record) for record in payload['trials']]

        return dataset

    def get_model(self, trial: int = -1):
        """Builds and returns a `BoTorchGP`, or a `DecoupledMOGP` with the 
        stored parameters in the dataset. Uses a torch state dict rather than
        refitting the model.
        """
        record = self.trials[trial]
        if record.state_dict is None:
            return None

        shared = {
            'noise'               : NoiseModel(**record.gp['noise']),
            'fit_hyperparameters' : False,
            'length_scale'        : record.gp['length_scale'],
            'signal_var'          : record.gp['signal_var'],
            'min_length_scale'    : record.gp['min_length_scale'],
            'state_dict'          : {key: torch.as_tensor(value, dtype=DTYPE)
                                     for key, value in record.state_dict.items()},
        }
        names = record.gp.get('objectives')
        if names is None:
            return BoTorchGP(objective=self.get_objective(trial), **shared)

        return DecoupledMOGP(
            objectives = DecoupledObjectives([
                Objective.from_record(record.measurements[name]) for name in names
            ]),
            **shared
        )

    def get_objective(self, trial: int = -1, name: str = None):
        """Uses the measurements to reconstruct the objective.

        Args:
            trial: the step to read.
            name: which objective. Defaults to the one that trial's GP was fit
                to, and to the first recorded before any fit.
        """
        record = self.trials[trial]
        if name is None:
            name = (record.gp or {}).get('objective') or next(iter(record.measurements))

        return Objective.from_record(record.measurements[name])

    def get_objectives(self, trial: int = -1):
        """Every objective at a trial, in one `DecoupledObjectives`."""
        return DecoupledObjectives([
            Objective.from_record(record)
            for record in self.trials[trial].measurements.values()
        ])

    def get_groundtruth(self):
        """The groundtruth the run was scored against rebuilt as a pypolar object.
        """
        if self.groundtruth is None:
            raise ValueError(f'{self.name} recorded no groundtruth')

        return self.groundtruth.build()

    def get_acquisition(self, bounds=None):
        """The acquisition the run queried, rebuilt from its arguments.

        The box is not stored with the saved params.
        """
        if self.acquisition is None:
            raise ValueError(f'{self.name} recorded no acquisition')

        return self.acquisition.build(bounds=bounds)

    def get_aux(self, name: str):
        """One auxiliary quantity across the run, (T, ...), NaN wherever the
        trial did not record it.

        The shape comes from the first trial that has the key, so a scalar
        stacks to (T,) and a (d,) recommendation to (T, d).
        """
        values = [trial.aux.get(name) for trial in self.trials]
        shape  = next((value.shape for value in values if value is not None), None)
        if shape is None:
            raise ValueError(f'no trial of {self.name} recorded {name!r}; it has '
                             f'{sorted(self.aux_names())}')

        return np.stack([np.full(shape, np.nan) if value is None else value
                         for value in values])

    def aux_names(self):
        """Every auxiliary key any trial recorded."""
        return {name for trial in self.trials for name in trial.aux}

    def get_actions(self):
        """(T, d) actions applied, in the order they were run."""
        actions = [trial.action for trial in self.trials if trial.action is not None]

        return np.vstack(actions) if actions else np.empty((0, 0))

    def get_action_labels(self):
        """One name per action dimension, in action order.

        Autofilled to `x0, x1, ...` if there are no action labels (legacy).

        Raises:
            ValueError: the run named a different number of dimensions than
                its trials recorded.
        """
        dim = _action_dim(self.trials)
        if self.action_labels is None:
            return [f'x{i}' for i in range(dim)]

        if dim and len(self.action_labels) != dim:
            raise ValueError(f'{self.name} names {len(self.action_labels)} action '
                             f'dimensions, but its trials record {dim}')

        return list(self.action_labels)

    def get_sources(self):
        """What chose each action, aligned with `get_actions`."""
        return [trial.source for trial in self.trials if trial.action is not None]
