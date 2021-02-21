import numpy as np
from torch.utils.data import Dataset

class OffPolicyMemory(Dataset):
    def __init__(self, capacity = 10000, datas = None):
        self.capacity   = capacity
        self.position   = 0

        if datas is None :
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32)

    def save_eps(self, state, action, reward, done, next_state):
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.dones.append(None)
            self.next_states.append(None)

        self.states[self.position]      = state
        self.actions[self.position]     = action
        self.rewards[self.position]     = reward
        self.dones[self.position]       = done
        self.next_states[self.position] = next_state

        self.position   = (self.position + 1) % self.capacity

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()

        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]