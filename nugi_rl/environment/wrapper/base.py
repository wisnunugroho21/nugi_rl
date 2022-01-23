import gym
import torch
from torch import device, Tensor

from nugi_rl.environment.base import Environment

class EnvWrapper(Environment):
    def __init__(self, env, agent_device: device):
        self.env            = env
        self.agent_device   = agent_device        

    def is_discrete(self):
        raise NotImplementedError

    def get_obs_dim(self):
        raise NotImplementedError
            
    def get_action_dim(self):
        raise NotImplementedError

    def reset(self) -> Tensor:
        state = self.env.reset()
        return torch.tensor(state).float().to(self.agent_device)

    def step(self, action: Tensor) -> tuple:
        action = action.squeeze().numpy()
        next_state, reward, done, info = self.env.step(action)

        next_state = torch.tensor(next_state).float().to(self.agent_device)
        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()