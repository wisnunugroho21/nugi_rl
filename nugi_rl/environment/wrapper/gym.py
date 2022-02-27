import gym
import torch
from torch import device, Tensor

from typing import Any, List, Tuple, Union

from nugi_rl.environment.base import Environment

class GymWrapper(Environment):
    def __init__(self, env, agent_device: device):
        self.env = env
        self.agent_device = agent_device        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is not gym.spaces.Box:
            return self.env.observation_space.n
        else:
            return self.env.observation_space.shape[0]
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self) -> Union[Tensor, List[Tensor]]:
        next_state = self.env.reset()

        if isinstance(next_state, list):
            for i in range(len(next_state)):
                next_state[i] = torch.tensor(next_state[i]).float().to(self.agent_device)
        else:
            next_state  = torch.tensor(next_state).float().to(self.agent_device)

        return next_state

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Any]:
        action = action.squeeze().cpu().numpy()
        next_state, reward, done, info = self.env.step(action)

        if isinstance(next_state, list):
            for i in range(len(next_state)):
                next_state[i] = torch.tensor(next_state[i]).float().to(self.agent_device)
        else:
            next_state  = torch.tensor(next_state).float().to(self.agent_device)

        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()