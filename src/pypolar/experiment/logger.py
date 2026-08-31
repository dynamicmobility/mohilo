"""One trial at a time: an action out to the device, values back from the
probes, into the objectives."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from pypolar.experiment.dataset import ExperimentDataset
from pypolar.experiment.probe import Probe
from pypolar.optimization.objectives import DecoupledObjectives, Objective


class Device(ABC):

    def send(self, action):
        """Sends an action to the device."""
        pass


class Logger:
    """A trial loop over decoupled objectives.

    Args:
        objectives: a `DecoupledObjectives`, or the objectives to wrap in one.
        probes: the instruments. Each names the objective it reports through
            `obj_name`, and several may report the same one.
        device: what an action is applied to.
        action_names: names of the action dimensions, which fix its length.
        aux_metrics: names of measurements recorded alongside the objectives.
    """

    def __init__(
        self,
        objectives  : DecoupledObjectives | list[Objective],
        probes      : list[Probe],
        # device      : Device,
        action_names: list[str] = None,
        aux_metrics : list[str] = None,
    ):
        if isinstance(objectives, list):
            objectives = DecoupledObjectives(list(objectives))
        
        self.objectives     = objectives
        self.action_names   = action_names
        self.aux_metrics    = aux_metrics
        # self.device         = device
        self.current_action = None

        self.probes = {probe.name: probe for probe in probes}
        if len(self.probes) != len(probes):
            raise ValueError(f'probe names must be unique, got '
                             f'{[probe.name for probe in probes]}')

        # one probe reports one objective; several probes may report one
        self.obj2probes: dict[str, list[Probe]] = {}
        for probe in probes:
            if probe.obj_name is None:
                continue
            if probe.obj_name not in self.objectives.names:
                raise ValueError(
                    f'probe {probe.name!r} reports {probe.obj_name!r}, which is '
                    f'not among the objectives {self.objectives.names}'
                )
            self.obj2probes.setdefault(probe.obj_name, []).append(probe)

    def begin_trial(self, action: np.ndarray, device_send_fn = None, args: dict = None,
                    kwargs: dict[str, dict] = None):
        """Applies an action and starts every probe measuring at it.

        Args:
            action: (d,) action, in raw units.
            args, kwargs: probe name -> the arguments that probe's caller takes.
        """
        if self.current_action is not None:
            raise RuntimeError('the last trial has not ended')

        args   = args or {}
        kwargs = kwargs or {}
        action = np.asarray(action, dtype=float).ravel()
        if self.action_names is not None and len(action) != len(self.action_names):
            raise ValueError(f'action has {len(action)} dimensions, expected '
                             f'{len(self.action_names)}: {self.action_names}')

        self.current_action = action
        # self.device.send(action)
        if device_send_fn:
            device_send_fn(action)

        for name, probe in self.probes.items():
            probe.measure(*args.get(name, ()), **kwargs.get(name, {}))

    @property
    def all_measurements_completed(self) -> bool:
        """Goes true when the current trial is done collecting measurements"""
        return all(probe.finished for probe in self.probes.values())
    
    def wait_for_measurements(self):
        while not self.all_measurements_completed:
            pass

    def end_trial(self, force=False):
        """Ends a trial, adding each probe's values to the objective it reports.

        Args:
            force: end even though some probe has not finished. What it did
                collect is kept, and an objective that got nothing is untouched.
        """
        if self.current_action is None:
            raise RuntimeError('no trial is open')

        if not force and not self.all_measurements_completed:
            unfinished = [name for name, probe in self.probes.items()
                          if not probe.finished]
            raise RuntimeError(
                f'these probes are still measuring: {unfinished}; pass '
                'force=True to end the trial with the values collected so far'
            )

        for probe in self.probes.values():
            probe.end_measurement_thread()

        for obj_name, probes in self.obj2probes.items():
            values = np.concatenate([probe.data for probe in probes])
            if not len(values):
                continue
            self.objectives.add_point(
                obj_name,
                np.tile(self.current_action, (len(values), 1)),
                values
            )

        self.current_action = None

    def resume(self, dataset: ExperimentDataset | Path | str):
        """Restores the measurements a saved run.
        Args:
            dataset: an `ExperimentDataset`, or a path to one `save` wrote.

        Returns:
            the number of trials whose action was applied, which is the index
            the loop picks up at. A run's closing record carries no action, so
            it does not count.

        Raises:
            RuntimeError: a trial is open.
            ValueError: the record's objectives are not the declared ones, or
                its actions have a different number of dimensions.
        """
        if self.current_action is not None:
            raise RuntimeError('a trial is open; end it before resuming')

        if not isinstance(dataset, ExperimentDataset):
            dataset = ExperimentDataset.load(Path(dataset))

        if not len(dataset):
            return 0

        measurements = dataset.trials[-1].measurements
        if set(measurements) != set(self.objectives.names):
            raise ValueError(f'{dataset.name} recorded {sorted(measurements)}, '
                             f'not the declared {sorted(self.objectives.names)}')

        restored = [Objective.from_record(measurements[name])
                    for name in self.objectives.names]

        if self.action_names is not None:
            dims = {obj.xdata.shape[1] for obj in restored if obj.xdata.size}
            if dims and dims != {len(self.action_names)}:
                raise ValueError(f'{dataset.name} recorded {dims} action '
                                 f'dimensions, expected {len(self.action_names)}: '
                                 f'{self.action_names}')

        self.objectives = DecoupledObjectives(restored)

        return len(dataset.get_actions())
