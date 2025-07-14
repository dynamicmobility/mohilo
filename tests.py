import numpy as np
import matplotlib.pyplot as plt

from pbl import PreferenceBasedLearning
from feedback import SimulatedFeedback
from gp import BasicGP
from sampler import RandomSampler

def test_get_idx():
    pbl = PreferenceBasedLearning(low=np.array([-1,-1]),
                                  high=np.array([1, 1]),
                                  action_dims=np.array([2, 2]),
                                  noise=0.01)
    print('ACTION SPACE:')
    print(pbl.action_space)
    print('LOW:', pbl.low)
    print('HIGH:', pbl.high)
    print('STEP', pbl.step)
    print()
    for action in pbl.action_space:
        print(f'Action: {action}')
        print(f'Index: {pbl.get_idx(action)}')
        print()
        
if __name__ == '__main__':
    print('Testing get_idx...')
    test_get_idx()