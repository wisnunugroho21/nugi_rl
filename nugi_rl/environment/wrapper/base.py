import torch
from gymnasium import Env
from torch import Tensor, device

from nugi_rl.environment.base import Environment


class EnvWrapper(Environment):
    def __init__(self, env: Env, agent_device: device) -> None:
        self.env = env
        self.agent_device = agent_device

    def is_discrete(self) -> bool:
        raise NotImplementedError

    def get_obs_dim(self) -> int:
        raise NotImplementedError

    def get_action_dim(self) -> int:
        raise NotImplementedError

    def reset(self) -> Tensor:
        state = self.env.reset()
        return torch.tensor(state).float().to(self.agent_device)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_np = action.squeeze().cpu().numpy()
        next_state, reward, done, truncated, info = self.env.step(action_np)

        next_state = torch.tensor(next_state).float().to(self.agent_device)
        reward = torch.tensor(reward).float().to(self.agent_device)
        done = torch.tensor(done).float().to(self.agent_device)

        return next_state, reward, done

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
