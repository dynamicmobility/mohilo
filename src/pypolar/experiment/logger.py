from collections.abc import Callable
import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import threading
import time
from abc import ABC, abstractmethod
from pypolar.optimization.objectives import Objective
import numpy as np

class Device(ABC):

    @abstractmethod
    def send(action):
        """Sends an action to the device."""

class Probe:
    """Callables standing in for instruments.
    """
    
    def __init__(
        self, 
        name    : str,
        caller  : Callable[[np.ndarray], float],
        repeats : int = 0,
        obj_name: str = None
    ):
        self.name     = name
        self.obj_name = obj_name
        self.caller   = caller
        self.repeats  = repeats
        self.finished = False
        
        self._data = None
        self.thread = None
        self.stop_event = None
        
    def call_and_detect(self, *args, **kwargs):
        while not self.stop_event.is_set() and not self.finished:
            d = self.caller(*args, **kwargs)
            self._data = d
            self.finished = True
        
        
    def measure(self, *args, **kwargs):
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.call_and_detect, args=args, kwargs=kwargs)
        self.thread.start()
        # self.thread.join()
        
    
    @property
    def data(self):
        if not self.finished:
            raise Exception('Not done yet')

        return self._data
    
    def end_measurement_thread(self):
        self.stop_event.set()
        self.thread.join()
    
    def reset(self):
        self.finished = False
        self._data = None
        if self.thread is not None:
            self.end_measurement_thread()
class Logger:
    
    def __init__(
        self,
        objectives: list[Objective],
        probes: list[Probe],
        action_names: list[str],
        device: Device,
        aux_metrics: list[str] = None,
    ):
        # self.objectives          = objectives
        self.objectives          = {obj.name: obj for obj in objectives}
        self.action_names        = action_names
        self.aux_metrics         = aux_metrics
        self.probes              = {probe.name: probe for probe in probes}
        self.obj2probe           = {probe.obj_name: probe for probe in probes if probe.obj_name is not None}
        self.device              = device
        self.current_action      = None
            
    def begin_trial(self, action: np.ndarray, args: dict = {}, kwargs: dict[str, dict] = {}):
        """Starts a trial with the given actions. Runs the probes to get measurements"""
        if self.current_action is not None:
            raise Exception('Last trial has not ended')
        
        self.current_action = action
        
        # reset the measurement devices
        for probe_name in self.probes:
            self.probes[probe_name].reset()

        # send the action
        self.device.send(action)
            
        # take measurements
        for probe_name in self.probes:
            probe = self.probes[probe_name]
            pargs, pkwargs = (), {}
            if probe.name in args:
                pargs = args[probe.name]
            elif probe.name in kwargs:
                pkwargs = kwargs[probe.name]
            probe.measure(*pargs, **pkwargs)
        
        
    @property
    def all_measurements_completed(self) -> bool:
        """Goes true with the current trial is done collecting measurements"""
        return len(self.probes) == sum([self.probes[probe_name].finished for probe_name in self.probes])
            
        
    def end_trial(self, force=False):
        """Ends a trial. If force=False, complains if some measurement
        were not completed"""
        
        for probe_name in self.probes:
            self.probes[probe_name].end_measurement_thread()
        
        for obj_name in self.obj2probe:
            self.objectives[obj_name].add_points(
                actions = self.current_action,
                values = self.obj2probe[obj_name].data
            )
        
        self.current_action = None
        
    def resume(self, experiment_path: Path):
        """Resumes an experiment from path"""
        pass
    

if __name__ == '__main__':
    def get_data_from_cart():
        time.sleep(2.0)
        return 0.0
    
    class Exo(Device):
        def __init__(self):
            pass
        
        def send(self, action):
            print('EXO GOT ACTION', action)
        
        
    metabolic_cart = Probe(
        name = 'Metabolic Cart',
        caller = get_data_from_cart,
        obj_name = 'cost'
    )
    
    metabolic_cost = Objective.from_empty(
        name = 'cost',
        maximize=False,
    )
    
    experiment = Logger(
        objectives   = [metabolic_cost],
        action_names = ['x', 'y', 'z'],
        device       = Exo(),
        probes       = [metabolic_cart]
    )
    action = np.ones(3)
    experiment.begin_trial(action)
    start = time.time()
    while not experiment.all_measurements_completed:
        print('Waiting...', time.time() - start)
        time.sleep(0.1)
    
    experiment.end_trial()
    print(experiment.objectives['cost'].ydata)
    print('done')
    metabolic_cart.reset()
    