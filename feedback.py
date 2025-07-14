import numpy as np


class SimulatedFeedback:
    
    def __init__(self, objective=None, action_space=None):
        """Simulates or experimentally collects feedback of a certain type"""
        
        self.truth = None
        self.objective = objective
        self.action_space = action_space
        
    def setup(self):
        pass
    
    def evaluate(self, curr, prev=None, get_pairwise=False, get_coactive=False, get_ordinal=False):
        """Evaluates the current and previous actions as if a perfect decider 
        was returning their preferences."""
        preference, coactive, ordinal = None, None, None
        
        # pairwise preference
        if get_pairwise:
            if prev is None:
                raise Exception('SimulatedFeedback.evaluate: curr and prev must both be specified for pairwise feedback')
            pref_lbl = self.objective(curr) >= self.objective(prev)
            preference = (curr, prev, pref_lbl)
        
        # coactive preference
        if get_coactive:
            if self.action_space is None:
                raise Exception('SimulatedFeedback.evaluate: action_space must not be None for coactive feedback simulation')
            a_bar = self.action_space[np.random.choice(len(self.action_space))]
            coac_lbl = self.objective(a_bar) >= self.objective(curr)
            coactive = (a_bar, curr, coac_lbl)
            
        # ordinal labels
        if get_ordinal:
            b0, b1 = self.objective.get_ordinal_label(curr)
            ordinal = (curr, b0, b1)
        
        return preference, coactive, ordinal
    
class AssistedFeedback:
    """LLM-assisted feedback collection"""
    pass # TODO

class UserFeedback:
    """User feedback collection"""
    pass # TODO
    
class SimulatedObjective:
    
    def __init__(self,
                 amplitude=1,
                 freq=2,
                 smooth=True,
                 ord_lbls=None,
                 action_space=None,
                 function='1D'):
        self.amp = amplitude
        self.freq = freq
        self.smooth = smooth
        self.ord_lbls = None
        if function == '1D':
            self.objective = self.objective_1d
        elif function == '2D':
            self.objective = self.objective_2d
        elif function == '3D':
            self.objective = self.objective_3d
        else:
            raise Exception('TODO')
        if ord_lbls:
            if action_space is None:
                raise Exception('SimulatedObjective1D: Action space required as \
                                 input for simulated ordinal labels.')
            f = self(action_space)
            self.ord_lbls = np.linspace(np.min(f), np.max(f), num=ord_lbls)
    
    def objective_1d(self, a):
        if self.smooth:
            return self.amp * np.exp(-a) * np.sin(self.freq * a)
        else:
            return self.amp * np.exp(-a) * (a if a % 4 < 2 else -a)
    
    def objective_2d(self, a):
        if len(a.shape) == 2:
            return 1 / (1 + a[:,0]**2) - a[:,1]**2
        return 1 / (1 + a[0]**2) - a[1]**2
    
    def objective_3d(self, a):
        if len(a.shape) == 2:
            return a[:,0]**2 + a[:,1]**2 + a[:,2]**2
        return a[0]**2 + a[1]**2 + a[2]**2

    def __call__(self, a):
        return self.objective(a)
        
    def get_ordinal_label(self, a):
        if self.ord_lbls is None:
            raise Exception('SimulatedObjective1D: Please define ord_lbls in class instantiation to use ordinal labels')
        b0 = -999
        for i in range(len(self.ord_lbls)):
            b1 = self.ord_lbls[i]
            if self(a) < b1:
                break
            b0 = b1
        if b0 == b1:
            b1 = 999
        return (b0, b1)