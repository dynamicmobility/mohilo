import numpy as np

class InternalReward:
    def __init__(self):
        raise NotImplementedError(
            "This is an abstract class."
        )
    
    def compute(self, x):
        raise NotImplementedError(
            "This is an abstract class."
        )

class IdealPoint(InternalReward):
    def __init__(self, w, delta, gamma=0.0):
        self.w = w
        self.delta = delta
        self.gamma = gamma
    
    def compute(self, x):  
        return -self.delta * np.sum(np.square(self.w - x), axis=-1) + self.gamma
    
    def __call__(self, x):
        return self.compute(x)
    
    
class NonStationaryIdealPoint(InternalReward):
    pass

class MultiObjectiveIdealPoint(InternalReward):
    def __init__(self, ideal_point_rewards: list[IdealPoint]):
        self.ideal_point_rewards = ideal_point_rewards

    def compute(self, x):
        fs = [ip_rew.compute(x) for ip_rew in self.ideal_point_rewards]
        return np.array(fs)
