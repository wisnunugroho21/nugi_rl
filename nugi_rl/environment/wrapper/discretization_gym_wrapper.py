import numpy as np
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper

class DiscretizationGymWrapper(GymWrapper):
    def __init__(self, env, bins = 15) -> None:
        super().__init__(env)

        self.act = np.arange(bins)
        self.act = (2 * self.act) / (bins - 1) - 1

    def step(self, action):
        action = self.act[action]
        return super().step(action)