import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class Config(BaseModel):
    # numpy arrays aren't pydantic-native types; allow them through as-is.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate(self):
        # Subclasses override with their own invariant checks.
        return self

    @classmethod
    def assert_equals_or_none(cls, atr, ref):
        """Asserts atr == ref OR ref is None"""
        assert (atr == ref) or (ref is None)


"""
PROBLEM CONFIGURATIONS
"""
class Regression(Config):
    action_low:  np.ndarray
    action_high: np.ndarray
    action_dims: np.ndarray
    precision:   float = 1.0

    def validate(self):
        assert np.all(self.action_high > self.action_low)
        assert np.issubdtype(self.action_dims.dtype, np.integer)
        assert self.precision > 0
        return self


class MultiObjectiveRegression(Config):
    action_low:  np.ndarray
    action_high: np.ndarray
    action_dims: np.ndarray
    precisions:  float | np.ndarray
    num_objs:    int = 2

    def validate(self):
        assert np.all(self.action_high > self.action_low)
        assert np.issubdtype(self.action_dims.dtype, np.integer)
        assert np.all(np.asarray(self.precisions) > 0)
        return self


class PBLConfig(Config):
    pass  # TODO


"""
OPTIMIZATION CONFIGURATIONS
"""
class GaussianProcess(Config):
    signal_variance:  float
    length_scale:     float
    kernel:           str = 'squared_exp'
    x0_init_method:   str = 'random'

    def validate(self):
        assert self.signal_variance > 0
        assert self.length_scale > 0
        return self


class MultiObjectiveGaussianProcess(Config):
    kernels:           list[str]
    x0_init_methods:   list[str]
    signal_variances:  list[float] | np.ndarray
    length_scales:     list[float] | np.ndarray
    num_objs:          int = 2

    def validate(self):
        assert np.all(np.asarray(self.signal_variances) > 0)
        assert np.all(np.asarray(self.length_scales) > 0)
        return self


"""
SAMPLING CONFIGURATIONS
"""
class DSTS(Config):
    rho: float = 0.5

    def validate(self):
        assert 0 < self.rho < 1
        return self


"""
ORACLE CONFIGURATION
"""
class IdealPoint(Config):
    w:     float | np.ndarray
    delta: float | np.ndarray
    gamma: float | np.ndarray


class BoundedIdealPoint(Config):
    w:            float | np.ndarray
    delta:        float | np.ndarray
    gamma:        float | np.ndarray
    lower_bound:  float | np.ndarray
    upper_bound:  float | np.ndarray

    def validate(self):
        assert np.all(self.upper_bound > self.lower_bound)
        return self


class NoisyRegressionOracle(Config):
    noise_std: float | np.ndarray

    def validate(self):
        assert np.all(np.asarray(self.noise_std) >= 0)
        return self


"""
HIGH-LEVEL CONFIGURATIONS
"""
class MOHILO(Config):
    problem:    MultiObjectiveRegression
    optimizer:  MultiObjectiveGaussianProcess
    sampler:    DSTS
    objective:  BoundedIdealPoint
    oracle:     NoisyRegressionOracle
    num_objs:   int
    save_dir:   str

    def validate(self):
        assert self.num_objs > 0
        self.assert_equals_or_none(self.num_objs, self.problem.num_objs)
        self.assert_equals_or_none(self.num_objs, self.optimizer.num_objs)
        return self
