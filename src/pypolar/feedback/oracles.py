import numpy as np
from pypolar.feedback import rewards

class PerfectOracle:
    """Simulates feedback from a perfect decider using a known objective function."""

    def __init__(self, objective=None, action_space=None):
        """
        Args:
            objective: a callable objective function (e.g. SimulatedObjective)
            action_space: the discretized action space (needed for coactive feedback)
        """
        self.objective = objective
        self.action_space = action_space

    def evaluate(self, curr, prev=None, get_pairwise=False, get_coactive=False, get_ordinal=False):
        """Evaluates the current and previous actions as if a perfect decider
        was returning their preferences.

        Args:
            curr: current action
            prev: previous action (required for pairwise feedback)
            get_pairwise: whether to return pairwise preference feedback
            get_coactive: whether to return coactive feedback
            get_ordinal: whether to return ordinal feedback

        Returns:
            Tuple of (preference, coactive, ordinal), each None if not requested.
        """
        preference, coactive, ordinal = None, None, None

        if get_pairwise:
            if prev is None:
                raise ValueError("curr and prev must both be specified for pairwise feedback")
            pref_lbl = bool(self.objective(curr) >= self.objective(prev))
            preference = (curr, prev, pref_lbl)

        if get_coactive:
            if self.action_space is None:
                raise ValueError("action_space must not be None for coactive feedback simulation")
            a_bar = self.action_space[np.random.choice(len(self.action_space))]
            coac_lbl = bool(self.objective(a_bar) >= self.objective(curr))
            coactive = (a_bar, curr, coac_lbl)

        if get_ordinal:
            b0, b1 = self.objective.get_ordinal_label(curr)
            ordinal = (curr, b0, b1)

        return preference, coactive, ordinal
    
    
class BradleyTerryOracle:
    def __init__(
        self, 
        beta_boltzmann, 
        reward_fn: rewards.InternalReward,
        rng: np.random.Generator
    ):
        self.beta_boltzmann = beta_boltzmann
        self.reward_fn = reward_fn
        self.rng = rng

    def compute_probabilities_from_reward(self, r_w, r_v):
        u_w = self.beta_boltzmann * r_w
        u_v = self.beta_boltzmann * r_v

        p_w = 1 / (1 + np.exp(u_v - u_w)) # numerical stability
        p_v = 1 - p_w

        return p_w, p_v
    
    def query(self, w, v):
        r_w = self.reward_fn.compute(w)
        r_v = self.reward_fn.compute(v)
        
        p_w, p_v = self.compute_probabilities_from_reward(r_w, r_v)
        idx = self.rng.choice([0, 1], p=[p_w, p_v])
        return idx
    

class MultiObjectiveOracle(BradleyTerryOracle):
    def __init__(
        self,
        beta_boltzmann    : float,
        reward_fn         : rewards.MultiObjectiveIdealPoint,
        rng               : np.random.Generator
    ): 
        super().__init__(beta_boltzmann, reward_fn, rng)
    
    def query(self, w: np.ndarray, v: np.ndarray):
        """
        say we have f1 and f2

        if w > v wrt f1 and w > v wrt f2, then prefer w
        if w > v wrt f1 and w < v wrt f2, then say that pref
        """
        
        # compute w on fs and v on fs
        # compute the probability of preferring w over v for each f
        w_fs, v_fs = self.get_groundtruth_rewards(w, v)
        
        p_ws, p_vs = self.compute_probabilities_from_reward(w_fs, v_fs)
        indexes = np.array(
            [self.rng.choice([0, 1], p=[p_w, 1 - p_w]) for p_w in p_ws]
        )
        if (indexes == 0).all():
            return 0
        elif (indexes == 1).all():
            return 1
        else:
            d_fs = np.abs(w_fs - v_fs)
            p_fs = np.exp(self.beta_boltzmann * d_fs) / np.sum(np.exp(self.beta_boltzmann * d_fs))
            chosen_f_idx = self.rng.choice(np.arange(len(w_fs)), p=p_fs)
            return indexes[chosen_f_idx]
        
    def mo_query(self, w: np.ndarray, v: np.ndarray):
        """Multi-objective query that returns a vector of preferences (one for 
        each objective function.)"""
        # compute w on fs and v on fs
        # compute the probability of preferring w over v for each f
        w_fs, v_fs = self.get_groundtruth_rewards(w, v)
        
        p_ws, p_vs = self.compute_probabilities_from_reward(w_fs, v_fs)
        indexes = np.array(
            [self.rng.choice([0, 1], p=[p_w, 1 - p_w]) for p_w in p_ws]
        )
        return indexes
        
    def get_groundtruth_rewards(self, w: np.ndarray, v: np.ndarray):
        return self.reward_fn.compute(w), self.reward_fn.compute(v)