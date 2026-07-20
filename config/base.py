import json
import types as _types
import typing
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


def _annotation_types(annotation):
    """Flatten a type annotation into its constituent types (unpacking unions)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(_types, 'UnionType', ()):
        out = []
        for arg in typing.get_args(annotation):
            out.extend(_annotation_types(arg))
        return out
    return [annotation]


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

    # ------------------------------------------------------------------ #
    # JSON (de)serialization                                             #
    # ------------------------------------------------------------------ #
    def to_jsonable_dict(self):
        """Return a JSON-safe dict (nested Configs and numpy arrays flattened)."""
        def convert(obj):
            if isinstance(obj, Config):
                return {name: convert(getattr(obj, name)) for name in type(obj).model_fields}
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj
        return convert(self)

    def to_json_string(self, **kwargs):
        """Serialize this config to a JSON string. Extra kwargs go to json.dumps."""
        return json.dumps(self.to_jsonable_dict(), **kwargs)

    def save_json_path(self, path):
        """Write this config to `path` as JSON. Returns the Path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_string(indent=2))
        return path

    @classmethod
    def from_jsonable_dict(cls, data):
        """Reconstruct a config from a plain dict (inverse of `to_jsonable_dict`)."""
        kwargs = {}
        for name, field in cls.model_fields.items():
            if name not in data:
                continue
            kwargs[name] = cls._decode_field(data[name], field.annotation)
        return cls(**kwargs)

    @classmethod
    def from_json_string(cls, s):
        """Reconstruct a config from a JSON string."""
        return cls.from_jsonable_dict(json.loads(s))

    @classmethod
    def load_json_path(cls, path):
        """Load a config from a JSON file at `path`."""
        return cls.from_jsonable_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def _decode_field(value, annotation):
        """Change a JSON value back to the type expected by a field annotation."""
        candidates = _annotation_types(annotation)
        # Nested Config models.
        if isinstance(value, dict):
            for t in candidates:
                if isinstance(t, type) and issubclass(t, Config):
                    return t.from_jsonable_dict(value)
        # Lists: keep as list if a list type is expected, else rebuild ndarray.
        if isinstance(value, list):
            expects_list = any(
                t is list or typing.get_origin(t) is list for t in candidates
            )
            if not expects_list and any(t is np.ndarray for t in candidates):
                return np.array(value)
        return value


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
