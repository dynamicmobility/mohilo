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
        elif function == '2D-circle':
            self.objective = self.objective_circle
        elif function == 'blue':
            self.objective = self.objective_blue
        elif function == 'dark':
            self.objective = self.objective_dark
        elif function == 'vibrant':
            self.objective = self.hsv_obj
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
    
    def objective_circle(self, a):
        arr = lambda i: a[i]
        r = 0.6
        xc = 0
        yc = 0
        if len(a.shape) == 2:
            arr = lambda i: a[:, i]
        return -np.abs(r**2 - ((arr(0) - xc)**2 + (arr(1) - yc)**2))
    
    def rgb2hsv(self, rgb):
        """
        Convert an array of RGB values to HSV.
        Input: rgb array of shape (..., 3) with values in [0, 255]
        Output: hsv array of shape (..., 3) with H in [0, 360], S and V in [0, 1]
        """
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        c_max = np.max(rgb, axis=-1)
        c_min = np.min(rgb, axis=-1)
        delta = c_max - c_min

        h = np.zeros_like(c_max)

        # Avoid division by zero
        mask = delta != 0

        # Red is max
        idx = (c_max == r) & mask
        h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx]) + 360) % 360

        # Green is max
        idx = (c_max == g) & mask
        h[idx] = (60 * ((b[idx] - r[idx]) / delta[idx]) + 120) % 360

        # Blue is max
        idx = (c_max == b) & mask
        h[idx] = (60 * ((r[idx] - g[idx]) / delta[idx]) + 240) % 360

        # Saturation
        s = np.zeros_like(c_max)
        s[c_max != 0] = delta[c_max != 0] / c_max[c_max != 0]

        # Value
        v = c_max

        hsv = np.stack([h, s, v], axis=-1)
        return hsv

    def hsv_obj(self, rgb):
        hsv = self.rgb2hsv(rgb)
        if len(hsv.shape) == 2:
            return -np.linalg.norm(hsv[:, 1:] - np.ones(2), axis=1)
        return -np.linalg.norm(hsv[1:] - np.ones(2))
    
    def objective_dark(self, rgb):
        if len(rgb.shape) == 2:
            return -np.linalg.norm(rgb, axis=1)
        return -np.linalg.norm(rgb)
    
    def objective_blue(self, a):
        if len(a.shape) == 2:
            return -np.linalg.norm([0, 0, 255] - a, axis=1)
        return -np.linalg.norm([0, 0, 255] - a)


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