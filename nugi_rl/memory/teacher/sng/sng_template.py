from copy import deepcopy
import torch
from torch.utils.data import Dataset

class SNGTemplateMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity

        self.states         = []
        self.goals          = []
        self.next_states    = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype = torch.float32), torch.tensor(self.goals[idx], dtype = torch.float32), torch.tensor(self.next_states[idx], dtype = torch.float32)

    def save_obs(self, state, goal, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.goals[0]
            del self.next_states[0]

        self.states.append(deepcopy(state))
        self.goals.append(deepcopy(goal))
        self.next_states.append(deepcopy(next_state))

    def save_replace_all(self, states, goals, next_states):
        self.clear_memory()
        self.save_all(states, goals, next_states)

    def save_all(self, states, goals, next_states):
        for state, goal, next_state in zip(states, goals, next_states):
            self.save_obs(state, goal, next_state)

    def get_all_items(self):         
        return self.states, self.goals, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            states      = self.states[start_position:end_position + 1]
            goals       = self.goals[start_position:end_position + 1]
            next_states = self.next_states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]
            goals       = self.goals[start_position:]
            next_states = self.next_states[start_position:]

        return states, goals, next_states 

    def clear_memory(self):
        del self.states[:]
        del self.goals[:]
        del self.next_states[:]

    def clear_idx(self, idx):
        del self.states[idx]
        del self.goals[idx]
        del self.next_states[idx]