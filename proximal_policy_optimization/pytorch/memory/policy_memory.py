import torch
import numpy as np
from torch.utils.data import Dataset

class PolicyMemory(Dataset):
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass     

    def save_eps(self, state, action, reward, done, next_state):
        pass           

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        pass

    def save_all(self, states, actions, rewards, dones, next_states):
        pass

    def get_all_items(self):         
        pass

    def clear_memory(self):
        pass