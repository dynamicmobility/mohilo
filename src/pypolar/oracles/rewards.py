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
    def __init__(self, w, delta):
        self.w = w
        self.delta = delta
    
    def compute(self, x):
        return -self.delta * np.sum(np.square(self.w - x))
    
    
class NonStationaryIdealPoint(InternalReward):
    pass

class MultiObjectiveIdealPoint(InternalReward):
    def __init__(self, ideal_point_rewards: list[IdealPoint]):
        self.ideal_point_rewards = ideal_point_rewards

    def compute(self, x):
        fs = [ip_rew.compute(x) for ip_rew in self.ideal_point_rewards]
        return np.array(fs)



class SimulatedObjective:
    """A callable simulated objective function for testing preference learning."""

    def __init__(
        self,
        amplitude=1,
        freq=2,
        smooth=True,
        ord_lbls=None,
        action_space=None,
        function="1D",
    ):
        """
        Args:
            amplitude: amplitude scaling for the objective
            freq: frequency parameter for the objective
            smooth: whether to use a smooth objective variant
            ord_lbls: number of ordinal label bins (None to disable)
            action_space: action space (required if ord_lbls is set)
            function: objective type ('1D', '2D', '3D', '2D-circle', 'blue', 'dark', 'vibrant')
        """
        self.amp = amplitude
        self.freq = freq
        self.smooth = smooth
        self.ord_lbls = None

        objectives = {
            "1D": self.objective_1d,
            "2D": self.objective_2d,
            "3D": self.objective_3d,
            "2D-circle": self.objective_circle,
            "blue": self.objective_blue,
            "dark": self.objective_dark,
            "vibrant": self.hsv_obj,
        }
        if function not in objectives:
            raise ValueError(f"Unknown objective function: {function}")
        self.objective = objectives[function]

        if ord_lbls:
            if action_space is None:
                raise ValueError("Action space required for simulated ordinal labels.")
            f = self(action_space)
            self.ord_lbls = np.linspace(np.min(f), np.max(f), num=ord_lbls)

    def objective_1d(self, a):
        if self.smooth:
            return self.amp * np.exp(-a) * np.sin(self.freq * a)
        else:
            return self.amp * np.exp(-a) * (a if a % 4 < 2 else -a)

    def objective_2d(self, a):
        if len(a.shape) == 2:
            return 1 / (1 + a[:, 0] ** 2) - a[:, 1] ** 2
        return 1 / (1 + a[0] ** 2) - a[1] ** 2

    def objective_3d(self, a):
        if len(a.shape) == 2:
            return a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2
        return a[0] ** 2 + a[1] ** 2 + a[2] ** 2

    def objective_circle(self, a):
        arr = lambda i: a[i]
        r = 0.6
        xc, yc = 0, 0
        if len(a.shape) == 2:
            arr = lambda i: a[:, i]
        return -np.abs(r**2 - ((arr(0) - xc) ** 2 + (arr(1) - yc) ** 2))

    def rgb2hsv(self, rgb):
        """Convert an array of RGB values to HSV.

        Args:
            rgb: array of shape (..., 3) with values in [0, 255]

        Returns:
            HSV array with H in [0, 360], S and V in [0, 1].
        """
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        c_max = np.max(rgb, axis=-1)
        c_min = np.min(rgb, axis=-1)
        delta = c_max - c_min

        h = np.zeros_like(c_max)
        mask = delta != 0

        idx = (c_max == r) & mask
        h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx]) + 360) % 360

        idx = (c_max == g) & mask
        h[idx] = (60 * ((b[idx] - r[idx]) / delta[idx]) + 120) % 360

        idx = (c_max == b) & mask
        h[idx] = (60 * ((r[idx] - g[idx]) / delta[idx]) + 240) % 360

        s = np.zeros_like(c_max)
        s[c_max != 0] = delta[c_max != 0] / c_max[c_max != 0]

        v = c_max
        return np.stack([h, s, v], axis=-1)

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
        """Get the ordinal label bucket for an action.

        Args:
            a: the action to label

        Returns:
            Tuple (b0, b1) representing the ordinal bucket bounds.
        """
        if self.ord_lbls is None:
            raise ValueError("Please define ord_lbls in class instantiation to use ordinal labels")
        b0 = -999
        for i in range(len(self.ord_lbls)):
            b1 = self.ord_lbls[i]
            if self(a) < b1:
                break
            b0 = b1
        if b0 == b1:
            b1 = 999
        return (b0, b1)
