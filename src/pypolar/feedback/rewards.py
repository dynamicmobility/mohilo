"""Commonly used reward functions for simulated feedback."""

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
