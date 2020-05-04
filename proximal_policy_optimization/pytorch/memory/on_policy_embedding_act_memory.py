import torch
import numpy as np
from torch.utils.data import Dataset

class Memory(Dataset):
    def __init__(self):
        self.actions = [] 
        self.states = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        self.available_actions = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32), np.array(self.available_actions[idx], dtype = np.float32)      

    def save_eps(self, state, reward, action, done, next_state, available_action):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state) 
        self.available_actions.append(available_action)

    def save_replace_all(self, states, rewards, actions, dones, next_states, available_actions):
        self.rewards = rewards
        self.states = states
        self.actions = actions
        self.dones = dones
        self.next_states = next_states    
        self.available_actions = available_actions

    def get_all_items(self):         
        return self.states, self.rewards, self.actions, self.dones, self.next_states, self.available_actions

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.available_actions[:]
