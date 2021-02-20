import numpy as np
from torch.utils.data import Dataset

class AuxMemory(Dataset):
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32)

    def save_eps(self, state):
        self.states.append(state) 

    def save_replace_all(self, states):
        self.states = states

    def save_all(self, states):
        self.states += states

    def get_all_items(self):         
        return self.states

    def clear_memory(self):
        del self.states[:]