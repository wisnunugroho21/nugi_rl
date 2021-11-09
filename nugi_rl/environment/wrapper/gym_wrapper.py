import gym
import torch
from torch import device, Tensor

from nugi_rl.environment.base import Environment

class GymWrapper(Environment):
    def __init__(self, env, agent_device: device):
        self.env = env
        self.agent_device = agent_device        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.env.observation_space.shape) > 1:                
                for i in range(len(self.env.observation_space.shape)):
                    state_dim *= self.env.observation_space.shape[i]            
            else:
                state_dim = self.env.observation_space.shape[0]

            return state_dim
        else:
            return self.env.observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self) -> Tensor:
        state = self.env.reset()
        return torch.tensor(state).float().to(self.agent_device)

    def step(self, action: Tensor) -> tuple:
        action = action.squeeze().tolist()
        next_state, reward, done, info = self.env.step(action)

        next_state = torch.tensor(next_state).float().to(self.agent_device)
        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()