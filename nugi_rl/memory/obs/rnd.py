from copy import deepcopy
import torch
from torch.utils.data import Dataset

class RndMemory(Dataset):
    def __init__(self, state_dim, capacity = 100000, device = torch.device('cuda:0')):
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

    def get_all_tensor(self):
        return torch.tensor(self.observations, dtype = torch.float32)

    def save_obs(self, obs):
        if len(self) >= self.capacity:
            del self.observations[0]

        self.observations.append(deepcopy(obs))

    def save_replace_all(self, observations):
        self.clear_memory()
        self.save_all(observations)

    def save_all(self, observations):
        for observation in observations:
            self.save_obs(observation)

    def get_all_items(self):
        return self.observations

    def get_all_tensor(self):
        return torch.tensor(self.observations, dtype = torch.float32)

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            observations    = self.observations[start_position:end_position + 1]
        else:
            observations    = self.observations[start_position:]

        return observations

    def clear_memory(self):
        del self.observations[:]

    def clear_idx(self, idx):
        del self.observations[idx]

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs           = mean_obs
        self.std_obs            = std_obs
        self.total_number_obs   = total_number_obs
        
    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards     = std_in_rewards
        self.total_number_rwd   = total_number_rwd
    