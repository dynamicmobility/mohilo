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
    def __init__(self, w, delta, gamma=0.0):
        self.w = w
        self.delta = delta
        self.gamma = gamma
    
    def compute(self, x):  
        return -self.delta * np.sum(np.square(self.w - x), axis=-1) + self.gamma
    
    
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
        self.w = w
        self.delta = delta
        self.gamma = gamma
        self.ip = IdealPoint(w, delta, gamma)
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def compute(self, x):  
        reward = self.ip.compute(x)
        normed = 0.5 * sigmoid(reward) + 0.5
        return (self.upper_bound - self.lower_bound) * normed + self.lower_bound
