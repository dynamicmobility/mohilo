import numpy as np


class RandomSampler:
    """Selects new actions using uniform random sampling."""

    def sample(self, actions):
        """Sample a random action from the action space.

        Args:
            actions: array of available actions

        Returns:
            A single randomly selected action.
        """
        idx = np.random.choice(a=actions.shape[0], replace=False)
        return actions[idx]
