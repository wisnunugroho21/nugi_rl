import torch
from torch import Tensor

from nugi_rl.memory.state.base import StateMemory


class RndMemory(StateMemory):
    def __init__(self, state_dim, device=torch.device("cuda"), capacity: int = 10000):
        self.capacity = capacity
        self.observations = []

        self.mean_obs = torch.zeros(state_dim, device=device)
        self.std_obs = torch.zeros(state_dim, device=device)

        self.std_in_rewards = torch.zeros(1, device=device)
        self.total_number_obs = torch.zeros(1, device=device)
        self.total_number_rwd = torch.zeros(1, device=device)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx) -> Tensor:
        return torch.tensor(self.observations[idx], dtype=torch.float32)

    def save(self, datas: Tensor) -> None:
        if len(self) >= self.capacity:
            del self.observations[0]

        self.observations.append(datas)

    def get(
        self, start_position: int | None = 0, end_position: int | None = None
    ) -> list[Tensor]:
        if end_position is not None and end_position != -1:
            if start_position is None or start_position < 0:
                states = self.observations[: end_position + 1]
            else:
                states = self.observations[start_position : end_position + 1]

        else:
            states = self.observations[start_position:]

        return states

    def clear(self, start_position: int = 0, end_position: int | None = None):
        if (
            start_position is not None
            and start_position > 0
            and end_position is not None
            and end_position != -1
        ):
            self.observations = [
                *self.observations[:start_position],
                *self.observations[end_position + 1 :],
            ]

        elif start_position is not None and start_position > 0:
            self.observations = self.observations[:start_position]

        elif end_position is not None and end_position != -1:
            self.observations = self.observations[end_position + 1 :]

        else:
            del self.observations
            self.observations = []

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs = mean_obs
        self.std_obs = std_obs
        self.total_number_obs = total_number_obs

    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards = std_in_rewards
        self.total_number_rwd = total_number_rwd
