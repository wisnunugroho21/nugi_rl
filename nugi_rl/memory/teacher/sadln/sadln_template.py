from copy import deepcopy
import torch
from torch.utils.data import Dataset

class SADLNTemplateMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity

        self.states         = []
        self.actions        = []
        self.logprobs       = []
        self.dones          = []
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype = torch.float32), torch.tensor(self.actions[idx], dtype = torch.float32), torch.tensor(self.logprobs[idx], dtype = torch.float32),\
            torch.tensor([self.dones[idx]], dtype = torch.float32), torch.tensor(self.next_states[idx], dtype = torch.float32)

    def save_obs(self, state, action, logprob, done, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.logprobs[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(deepcopy(state))
        self.actions.append(deepcopy(action))
        self.logprobs.append(deepcopy(logprob))
        self.dones.append(done)
        self.next_states.append(deepcopy(next_state))

    def save_replace_all(self, states, actions, logprobs, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, logprobs, dones, next_states)

    def save_all(self, states, actions, logprobs, dones, next_states):
        for state, action, logprobs, dones, next_states in zip(states, actions, logprobs, dones, next_states):
            self.save_obs(state, action, logprobs, dones, next_states)

    def get_all_items(self):         
        return self.states, self.actions, self.logprobs, self.dones, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            states      = self.states[start_position:end_position + 1]
            actions     = self.actions[start_position:end_position + 1]
            logprobs    = self.logprobs[start_position:end_position + 1]
            dones       = self.dones[start_position:end_position + 1]
            next_states = self.next_states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]
            actions     = self.actions[start_position:]
            logprobs    = self.logprobs[start_position:]
            dones       = self.dones[start_position:]
            next_states = self.next_states[start_position:]

        return states, actions, logprobs, dones, next_states 

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.dones[:]
        del self.next_states[:]

    def clear_idx(self, idx):
        del self.states[idx]
        del self.actions[idx]
        del self.logprobs[idx]
        del self.dones[idx]
        del self.next_states[idx]