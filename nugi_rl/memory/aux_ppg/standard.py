from copy import deepcopy
import torch
from torch.utils.data import Dataset

class AuxPpgMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity

        if datas is None:
            self.states         = []
        else:
            self.states = datas
            if len(self) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx])

    def save_obs(self, state):
        if len(self) >= self.capacity:
            del self.states[0]

        self.states.append(deepcopy(state))

    def save_replace_all(self, states):
        self.clear_memory()
        self.save_all(states)

    def save_all(self, states):
        for state in states:
            self.save_obs(state)

    def get_all_items(self):         
        return self.states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            states      = self.states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]

        return states

    def clear_memory(self):
        del self.states[:]

    def clear_idx(self, idx):
        del self.states[idx]