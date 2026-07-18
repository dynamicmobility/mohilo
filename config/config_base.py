from dataclasses import dataclass
import numpy as np

"""
PROBLEM CONFIGURATIONS
"""
@dataclass
class ProblemConfig:
    action_low:  np.ndarray[np.float32] = np.array(0.0)
    action_high: np.ndarray[np.float32] = np.array(4.0)
    action_dims: np.ndarray[np.float32] = np.array(100)


@dataclass
class RegressionConfig(ProblemConfig):
    precision: 1e1

@dataclass
class PBLConfig(ProblemConfig):
    pass # TODO


"""
OPTIMIZATION CONFIGURATIONS
"""
@dataclass
class GaussianProcessConfig:
    kernel            : str = 'squared_exp'
    signal_variance   : float = 10.0
    length_scale      : float = 1.0
    x0_init_method    : str = 'random'


"""
SAMPLING CONFIGURATIONS
"""
@dataclass
class DSTSConfig:
    rho: float = 0.5


"""
ORACLE CONFIGURATION
"""
class IdealPointConfig:
    w:     float = 1.0
    delta: float = 1.0
    gamma: float = 0.0


class BoundedIdealPointConfig:
    lower_bound: float = 0.0
    upper_bound: float = 1.0


class NoisyRegressionOracle:
    noise_std: int = 0.5

"""
HIGH-LEVEL CONFIGURATIONS
"""
@dataclass
class MOHILOConfig:
    problems: list[ProblemConfig]
    optimizer: list[GaussianProcessConfig]
    sampler: DSTSConfig
    objectives: list[BoundedIdealPointConfig]



HipExoSimConfig = MOHILOConfig()