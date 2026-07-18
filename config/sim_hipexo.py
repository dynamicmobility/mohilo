from dataclasses import dataclass
import numpy as np

@dataclass
class ProblemConfig:
    action_low:  int = np.array(0.0)
    action_high: int = np.array(4.0)
    action_dims: int = np.array(100)


@dataclass
class RegressionConfig(ProblemConfig):
    precision: 1e1

@dataclass
class PBLConfig(ProblemConfig):
    pass


@dataclass
class OptimizationConfig:
    pass

@dataclass
class GaussianProcessConfig:
    pass

@dataclass
class HILOConfig:
    problems: list[ProblemConfig]


HipExoSimMOHILO