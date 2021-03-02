import numpy as np
from torch.utils.data import Dataset

class ClrMemory(Dataset):
    def __init__(self, capacity = 10000):
        self.capacity   = capacity
        self.states     = []
        self.position   = 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32)

    def save_eps(self, state):
        if len(self.states) < self.capacity:
            self.states.append(None)

        self.states[self.position]  = state
        self.position               = (self.position + 1) % self.capacity

    def save_replace_all(self, states):
        self.clear_memory()

        for state in states:
            self.save_eps(state)

    def save_all(self, states):
        for state in states:
            self.save_eps(state)

    def get_all_items(self):         
        return self.states

    def clear_memory(self):
        del self.states[:]