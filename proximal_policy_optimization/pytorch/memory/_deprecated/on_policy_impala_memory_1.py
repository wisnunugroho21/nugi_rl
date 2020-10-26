import torch
import numpy as np
from torch.utils.data import Dataset

class OnMemory(Dataset):
    def __init__(self, datas = None):
        if datas is None :
            self.states             = []
            self.actions            = []
            self.rewards            = []
            self.dones              = []
            self.next_states        = []
            self.logprobs           = []
            self.next_next_states   = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs, self.next_next_states = datas

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), np.array(self.logprobs[idx], dtype = np.float32), \
            np.array(self.next_next_states[idx], dtype = np.float32)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs, self.next_next_states
    
    def pop(self, idx):
        return self.states.pop(idx), self.actions.pop(idx), self.rewards.pop(idx), self.dones.pop(idx), self.next_states.pop(idx), \
            self.logprobs.pop(idx), self.next_next_states.pop(idx)
    
    def save_eps(self, state, action, reward, done, next_state, logprob, next_next_state = None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.logprobs.append(logprob)

        if next_next_state is not None:
            self.next_next_states.append(next_state)            

    def save_replace_all(self, states, actions, rewards, dones, next_states, logprobs, next_next_states):
        self.states         = states
        self.actions        = actions
        self.rewards        = rewards
        self.dones          = dones
        self.logprobs       = logprobs

        if next_next_states is not None:
            self.next_next_states   = next_next_states

    
    def clear_memory(self, idx = -100):
        if idx == -100:
            del self.states[:]
            del self.actions[:]
            del self.rewards[:]
            del self.dones[:]
            del self.next_states[:]
            del self.logprobs[:]
            del self.next_next_states[:]

        else:
            del self.states[idx]
            del self.actions[idx]
            del self.rewards[idx]
            del self.dones[idx]
            del self.next_states[idx]
            del self.logprobs[idx]
            del self.next_next_states[idx]        

    def convert_next_states_to_next_next_states(self):
        next_next_states    = []
        length              = len(self.next_states)

        for i in range(1, length - 1):
            next_next_states.append(self.next_states[i + 1])

        next_next_states.append(self.next_states[length - 1])
        self.next_next_states = next_next_states

