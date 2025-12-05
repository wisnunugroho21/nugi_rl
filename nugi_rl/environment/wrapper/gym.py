import gymnasium as gym
import numpy as np
import torch
from gymnasium import Env
from gymnasium.wrappers import RescaleAction, RescaleObservation
from torch import Tensor, device

from nugi_rl.environment.base import Environment


class GymWrapper(Environment):
    def __init__(self, env: Env, agent_device: device) -> None:
        obs_shape = np.array(env.observation_space.shape)
        min_obs = np.array([-1.0])
        max_obs = np.array([1.0])

        self.env = RescaleObservation(
            env, np.repeat(min_obs, obs_shape), np.repeat(max_obs, obs_shape)
        )

        if type(env.action_space) is gym.spaces.Discrete:
            act_shape = np.array(env.action_space.shape)
            min_act = np.array([-1.0])
            max_act = np.array([1.0])

            self.env = RescaleAction(
                self.env, np.repeat(min_act, act_shape), np.repeat(max_act, act_shape)
            )

        self.agent_device = agent_device

    def is_discrete(self) -> bool:
        return type(self.env.action_space) is gym.spaces.Discrete

    def get_obs_dim(self) -> int:
        if self.env.observation_space.shape is None:
            return 0

        return self.env.observation_space.shape[0]

    def get_action_dim(self) -> int:
        if self.env.action_space.shape is None:
            return 0

        if type(self.env.action_space) is gym.spaces.Discrete:
            return self.env.action_space.n

        return self.env.action_space.shape[0]

    def reset(self) -> Tensor:
        next_state, _ = self.env.reset()

        if isinstance(next_state, list):
            next_state_tensor = torch.stack(
                [torch.tensor(ns).float().to(self.agent_device) for ns in next_state]
            )
        else:
            next_state_tensor = torch.tensor(next_state).float().to(self.agent_device)

        return next_state_tensor

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_np = action.cpu().numpy()
        next_state, reward, done, truncated, _ = self.env.step(action_np)

        if isinstance(next_state, list):
            next_state_tensor = torch.stack(
                [torch.tensor(ns).float().to(self.agent_device) for ns in next_state]
            )
        else:
            next_state_tensor = torch.tensor(next_state).float().to(self.agent_device)

        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done or truncated).float().to(self.agent_device)

        return next_state_tensor, reward, done

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
