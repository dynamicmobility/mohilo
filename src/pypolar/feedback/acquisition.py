from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
    AcquisitionFunction
)
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import (
    SyntheticTestFunction
)

from pypolar.optimization.gp import BoTorchGP


class AcquisitionFunction:
    
    def __init__(self, acqf: AcquisitionFunction, bounds, num_restarts, raw_samples):
        pass
    
    def query(self, model: BoTorchGP, q, bounds=None):
        pass