import numpy as np

class RandomSampler: # this is also sort of the dataset
    
    def __init__(self):
        """Selects new actions using random sampling"""
        
        pass
    
    def sample(self, actions):
        idx = np.random.choice(a=actions.shape[0], replace=False)
        return actions[idx]
    
class ThompsonSampler:
    pass # TODO

class UCB1Sampler:
    pass # TODO

