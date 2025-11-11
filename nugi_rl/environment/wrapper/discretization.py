import gymnasium as gym
import torch
from gymnasium import Env
from torch import Tensor, device

from nugi_rl.environment.base import Environment


class DiscretizationWrapper(Environment):
    def __init__(
        self, env: Env, agent_device: device, bins: int = 15, act_dim: int | None = None
    ) -> None:
        self.env = env
        self.bins = bins
        self.agent_device = agent_device

        self.act = torch.arange(bins, device=agent_device)
        self.act = (2 * self.act) / (bins - 1) - 1

        if act_dim is None:
            self.act_dim = self.get_action_dim()
        else:
            self.act_dim = act_dim

    def is_discrete(self) -> bool:
        return type(self.env.action_space) is gym.spaces.Discrete

    def get_obs_dim(self) -> int:
        if (self.env.observation_space.shape is None)
            return 0

        return self.env.observation_space.shape[0]

    def get_action_dim(self) -> int:
        if (self.env.action_space.shape is None)
            return 0

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
        action = action.flatten().long()
        action = self.act[action].reshape(-1, self.act_dim)

        next_state, reward, done, truncated, info = self.env.step(action)

        if isinstance(next_state, list):
            next_state_tensor = torch.stack(
                [torch.tensor(ns).float().to(self.agent_device) for ns in next_state]
            )
        else:
            next_state_tensor = torch.tensor(next_state).float().to(self.agent_device)

        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state_tensor, reward, done

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
