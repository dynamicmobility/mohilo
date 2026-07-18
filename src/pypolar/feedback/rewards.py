"""Commonly used reward functions for simulated feedback."""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class InternalReward:
    def __init__(self):
        raise NotImplementedError(
            "This is an abstract class."
        )
    
    def compute(self, x):
        raise NotImplementedError(
            "This is an abstract class."
        )
        
    def __call__(self, x):
        return self.compute(x)

class IdealPoint(InternalReward):
    def __init__(self, w: np.ndarray, delta, gamma=0.0):
        self.w = w
        self.delta = delta
        self.gamma = gamma
    
    def compute(self, x):
        if type(x) == float:
            return -self.delta * np.sum(np.square(self.w - x), axis=0) + self.gamma
        elif len(x.shape) == 1:
            x = x.reshape((-1, 1))
            return -self.delta * np.sum(np.square(self.w - x), axis=1) + self.gamma
        elif len(x.shape) == 2:
            assert 1 in x.shape
            if x.shape[0] == 1:
                x = x.T
        else:
            raise Exception('Incompatible shape')
        
        """Make this multi objective? or many rewards?"""
        
        return -self.delta.reshape(-1, 1) * np.sum(np.square(self.w - x), axis=1) + self.gamma.reshape(-1, 1)
        # return ((x[:, None, :] - self.w[None, :, :]) ** 2).sum(axis=-1)
    
    
class NonStationaryIdealPoint(InternalReward):
    pass

class MultiObjectiveIdealPoint(InternalReward):
    def __init__(self, ideal_point_rewards: list[IdealPoint]):
        self.ideal_point_rewards = ideal_point_rewards

    def compute(self, x):
        fs = [ip_rew.compute(x) for ip_rew in self.ideal_point_rewards]
        return np.array(fs)
    


class BoundedIdealPoint(InternalReward):

    def __init__(self, w, delta, gamma=0.0, lower_bound=-1.0, upper_bound=1.0):
        w = np.asarray(w, dtype=float)
        if w.ndim == 1:
            # single objective: treat as a stack of one ideal point
            w = w[None, :]
        elif w.ndim != 2:
            raise ValueError(
                f"w must be a 1D vector or a 2D (m, d) stack; got shape {w.shape}"
            )
        self.w = w                      # (m, d)
        self.m, self.d = w.shape

        self.delta = np.asarray(delta, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.lower_bound = np.asarray(lower_bound, dtype=float)
        self.upper_bound = np.asarray(upper_bound, dtype=float)

    def _ideal_point(self, x):
        """Raw (unbounded) objective-wise ideal-point reward.

        Returns ``(m,)`` for a single vector ``x`` and ``(N, m)`` for a batch.
        Per-objective params (length ``m``) broadcast along the trailing axis.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1)            # scalar action, d == 1

        if x.ndim == 1:
            # x: (d,) -> sq: (m,)
            sq = np.sum(np.square(self.w - x), axis=1)
        elif x.ndim == 2:
            # x: (N, d) -> sq: (N, m)
            diff = x[:, None, :] - self.w[None, :, :]   # (N, m, d)
            sq = np.sum(np.square(diff), axis=2)        # (N, m)
        else:
            raise ValueError(
                f"x must be a (d,) vector or an (N, d) batch; got shape {x.shape}"
            )

        return -self.delta * sq + self.gamma

    def compute(self, x):
        reward = self._ideal_point(x)
        normed = sigmoid(reward)
        return (self.upper_bound - self.lower_bound) * normed + self.lower_bound
