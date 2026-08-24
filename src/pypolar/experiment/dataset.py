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
from pypolar.feedback.acquisition import AcquisitionFunction, acquisition_factory_1d
from pypolar.feedback.synthetic import make_synthetic
from pypolar.optimization.gp import DTYPE, BoTorchGP, NoiseModel
from pypolar.optimization.objectives import DecoupledObjectives, Objective


def _gp_record(gp: BoTorchGP):
    """The arguments a GP was built with, read off the GP itself."""
    return {
        'objective'        : gp.objective.name,
        'noise'            : asdict(gp.noise),
        'length_scale'     : gp.length_scale,
        'signal_var'       : gp.signal_var,
        'min_length_scale' : gp.min_length_scale,
    }


def _trial_from_json(record):
    """One trial read back, with its arrays as arrays again."""
    for key in ('action', 'recommended'):
        if record[key] is not None:
            record[key] = np.asarray(record[key], dtype=float)

    for measurement in record['measurements'].values():
        measurement['xdata'] = np.asarray(measurement['xdata'], dtype=float)
        measurement['ydata'] = np.asarray(measurement['ydata'], dtype=float)

    return TrialDataset(**record)


@dataclass
class TrialDataset:
    """One step of a run: the measurements in hand, the GP fit to them, and the
    action that GP chose next.

    A record is the state *before* its action was applied, so trial t's
    measurements are what trial t-1's action produced. The last record of a run
    carries no action -- nothing further was chosen -- and holds the fit to
    every measurement taken.

    Attributes:
        trial: the step's index.
        measurements: objective name -> that objective's `to_record`.
        acquisition: the arguments `acquisition_factory_1d` was called with.
        groundtruths: objective name -> the arguments `make_synthetic` used.
        action: (d,) action chosen here, in raw units, or None on the last record.
        source: what chose it -- an acquisition's name, or 'random'.
        state_dict: the fitted GP's tensors, as lists. None before the first fit.
        gp: the arguments that GP was built with.
        recommended: (d,) action the GP's posterior mean peaks at, or None.
        regret: the groundtruth gap at `recommended`, in spreads, or None.
    """

    trial        : int
    measurements : dict[str, dict]
    acquisition  : dict
    groundtruths : dict[str, dict]  # make this optional (real study wont have this)
    action       : np.ndarray | None      = None
    source       : str | None             = None
    state_dict   : dict[str, list] | None = None
    gp           : dict | None            = None
    recommended  : np.ndarray | None      = None
    regret       : float | None           = None


@dataclass
class ExperimentDataset:
    """One run, saved as a single json file. Includes model, data, acquisition.

    Attributes:
        name: what the run is called.
        acquisition, groundtruths: as on `TrialDataset`, stamped into each.
        config: the constants the run was made under, hashed on save so two
            runs that differ in configuration cannot be mistaken for one.
        trials: the run, one record per step.
        path: where it was last written, or read from.
    """
    # TODO: when regret is not available, warn/error if requested
    name         : str
    acquisition  : dict               = field(default_factory=dict)
    groundtruths : dict[str, dict]    = field(default_factory=dict)
    config       : dict               = field(default_factory=dict)
    trials       : list[TrialDataset] = field(default_factory=list)
    path         : Path | None        = None

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, trial):
        return self.trials[trial]

    def add_trial(self, objectives, gp=None, action=None, source=None,
                  recommended=None, regret=None):
        """Records one step, and returns the `TrialDataset` it appended.

        Args:
            objectives: the `DecoupledObjectives` as they stand *before* the
                action is applied.
            gp: the `BoTorchGP` fit to them, or None when none was fit yet.
            action, source, recommended, regret: as on `TrialDataset`.
        """
        record = TrialDataset(
            trial        = len(self.trials),
            measurements = {name: objectives[name].to_record()
                            for name in objectives.names},
            acquisition  = self.acquisition,
            groundtruths = self.groundtruths,
            action       = None if action is None
                           else np.asarray(action, dtype=float).ravel(),
            source       = source,
            state_dict   = None if gp is None
                           else {key: value.tolist()
                                 for key, value in gp.model.state_dict().items()},
            gp           = None if gp is None else _gp_record(gp),
            recommended  = None if recommended is None
                           else np.asarray(recommended, dtype=float).ravel(),
            regret       = None if regret is None else float(regret),
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
            'name'         : self.name,
            'fingerprint'  : fingerprint(self.config),
            'config'       : self.config,
            'acquisition'  : self.acquisition,
            'groundtruths' : self.groundtruths,
            'trials'       : [asdict(trial) for trial in self.trials],
        }
        # a state dict's constraint buffers are infinite, which python's json
        # writes as `Infinity` and reads back; no stricter reader is promised
        self.path.write_text(json.dumps(jsonable(payload), indent=2))

        return self.path

    @classmethod
    def load(cls, path: Path):
        """A run read back from `save`."""
        path    = Path(path)
        payload = json.loads(path.read_text())
        dataset = cls(
            name         = payload['name'],
            acquisition  = payload['acquisition'],
            groundtruths = payload['groundtruths'],
            config       = payload['config'],
            path         = path,
        )
        dataset.trials = [_trial_from_json(record) for record in payload['trials']]

        return dataset

    def get_model(self, trial: int = -1):
        """Makes a `BoTorchGP` with the stored parameters and returns it.

        The state dict *is* the fit, so nothing is refit: the same measurements
        under the same configuration, with the same tensors loaded over them,
        give back the posterior the run actually queried. None before the first
        fit.
        """
        record = self.trials[trial]
        if record.state_dict is None:
            return None

        return BoTorchGP(
            objective           = self.get_objective(trial),
            noise               = NoiseModel(**record.gp['noise']),
            fit_hyperparameters = False,
            length_scale        = record.gp['length_scale'],
            signal_var          = record.gp['signal_var'],
            min_length_scale    = record.gp['min_length_scale'],
            # json numbers are decimal and torch infers float32 from python
            # floats, so without the cast the fit comes back at half its precision
            state_dict          = {key: torch.as_tensor(value, dtype=DTYPE)
                                   for key, value in record.state_dict.items()},
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

    def get_groundtruth(self, trial: int = -1):
        """The groundtruth functions a trial ran against, by objective name.

        Rebuilt from their arguments, so the functions are identical; their
        noise streams restart from the seed rather than resuming.
        """
        return {name: make_synthetic(**spec)
                for name, spec in self.trials[trial].groundtruths.items()}

    def get_acquisition(self, trial: int = -1):
        """The acquisition a trial queried, over that trial's objective."""
        record = self.trials[trial]

        return AcquisitionFunction(
            acqf      = acquisition_factory_1d(**record.acquisition),
            objective = self.get_objective(trial),
        )

    def get_regret(self):
        """Regret from all trials, (T,), NaN wherever no GP was fit."""
        return np.array([np.nan if trial.regret is None else trial.regret
                         for trial in self.trials])

    def get_actions(self):
        """(T, d) actions applied, in the order they were run."""
        actions = [trial.action for trial in self.trials if trial.action is not None]

        return np.vstack(actions) if actions else np.empty((0, 0))

    def get_sources(self):
        """What chose each action, aligned with `get_actions`."""
        return [trial.source for trial in self.trials if trial.action is not None]

    def get_recommendations(self):
        """(T, d) action each trial's GP would have recommended, NaN before the
        first fit."""
        if not self.trials:
            return np.empty((0, 0))

        # the width comes from the records rather than from the config, which
        # holds whatever the run chose to stamp there
        width = next((len(trial.recommended) for trial in self.trials
                      if trial.recommended is not None), 0)

        return np.vstack([np.full(width, np.nan) if trial.recommended is None
                          else trial.recommended for trial in self.trials])
