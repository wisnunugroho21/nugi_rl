import torch
from copy import deepcopy

class RndMemory():
    def __init__(self, state_dim, device = torch.device('cuda'), capacity: int = 10000):        
        self.capacity           = capacity
        self.observations       = []

        self.mean_obs           = torch.zeros(state_dim).to(device)
        self.std_obs            = torch.zeros(state_dim).to(device)

        self.std_in_rewards     = torch.zeros(1).to(device)
        self.total_number_obs   = torch.zeros(1).to(device)
        self.total_number_rwd   = torch.zeros(1).to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return torch.tensor(self.observations[idx], dtype = torch.float32)

    def save(self, obs) -> None:
        if len(self) >= self.capacity:
            del self.observations[0]

        self.observations.append(deepcopy(obs))

    def get(self, start_position: int = 0, end_position: int = None) -> tuple:
        if end_position is not None or end_position == -1:
            obs  = self.observations[start_position:end_position + 1]
        else:
            obs  = self.observations[start_position:]

        return obs

    def save_all(self, obs) -> None:
        for ob in obs:
            self.save(ob)

    def clear(self, start_position: int = 0, end_position: int = None):
        if end_position is not None and end_position != -1:
            del self.observations[start_position : end_position + 1]
        else:
            del self.observations[start_position :]

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs           = mean_obs
        self.std_obs            = std_obs
        self.total_number_obs   = total_number_obs
        
    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards     = std_in_rewards
        self.total_number_rwd   = total_number_rwd