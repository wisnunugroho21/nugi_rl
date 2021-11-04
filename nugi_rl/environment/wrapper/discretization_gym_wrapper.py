import torch
import numpy as np
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from torch import device, Tensor

class DiscretizationGymWrapper(GymWrapper):
    def __init__(self, env, agent_device: device, bins: int = 15) -> None:
        super().__init__(env, agent_device)

        self.act = np.arange(bins)
        self.act = (2 * self.act) / (bins - 1) - 1

    def step(self, action: Tensor) -> tuple:
        action = action.tolist()
        action = self.act[action]
        
        next_state, reward, done, info = self.env.step(action)

        next_state = torch.tensor(next_state).float().to(self.agent_device)
        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state, reward, done, info