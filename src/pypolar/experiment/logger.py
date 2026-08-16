import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, StorageConfig
from pypolar.experiment.probe import Probe
import threading

class Logger:
    
    def __init__(
        self,
        columns: list[str],
        action_names: list[str],
        storage_config: StorageConfig = None, # TODO: look into this
        measurements: Probe = None
    ):
        raise NotImplementedError()
    
    def begin_trial(self, actions):
        """Starts a trial with the given actions"""
        
    @property
    def measurements_completed(self) -> bool:
        """Goes true with the current trial is done collecting measurements"""
        
    def end_trial(self, force=False):
        """Ends a trial. If force=False, complains if some measurement
        were not completed"""
        
    def resume(self, experiment_path: Path):
        """Resumes an experiment from path"""
        pass
    
    