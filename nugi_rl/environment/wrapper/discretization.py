import torch
from torch import device, Tensor

from nugi_rl.environment.base import Environment

class DiscretizationWrapper(Environment):
    def __init__(self, env, agent_device: device, bins: int = 15, act_dim: int = None) -> None:
        super().__init__(env, agent_device)

        self.env        = env
        self.bins       = bins
        self.act_dim    = act_dim

        self.act        = torch.arange(bins, device = agent_device)
        self.act        = (2 * self.act) / (bins - 1) - 1       
        
        if act_dim is None:
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