import numpy as np
from pypolar import RandomSampler


class TestRandomSampler:
    def test_sample_returns_action(self):
        sampler = RandomSampler()
        actions = np.array([[0, 0], [1, 1], [2, 2]])
        action = sampler.sample(actions)
        assert action.shape == (2,)

    def test_sample_is_from_action_space(self):
        sampler = RandomSampler()
        actions = np.array([[0], [1], [2], [3]])
        for _ in range(20):
            action = sampler.sample(actions)
            assert action in actions
