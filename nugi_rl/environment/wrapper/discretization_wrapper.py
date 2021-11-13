import torch
import numpy as np
from nugi_rl.environment.wrapper.gym_wrapper import GymWrapper
from torch import device, Tensor

class DiscretizationWrapper(GymWrapper):
    def __init__(self, env, agent_device: device, bins: int = 15) -> None:
        super().__init__(env, agent_device)

        self.act = torch.arange(bins, device = agent_device)
        self.act = (2 * self.act) / (bins - 1) - 1
        
        self.env = env
        self.bins = bins
        self.act_dim = self.env.get_action_dim()

    def step(self, action: Tensor) -> Tensor:
        action = action.flatten().long()
        action = self.act[action].reshape(-1, self.act_dim)
        
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def is_discrete(self):
        return self.env.is_discrete()

    def get_obs_dim(self):
        return self.env.get_obs_dim()
            
    def get_action_dim(self):
        return self.env.get_action_dim()