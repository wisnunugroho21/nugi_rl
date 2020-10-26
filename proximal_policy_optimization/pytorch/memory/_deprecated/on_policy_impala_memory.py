import torch
import numpy as np
from torch.utils.data import Dataset

class OnMemoryImpala(Dataset):
    def __init__(self, datas = None):
        if datas is None :
            self.states             = []
            self.actions            = []
            self.rewards            = []
            self.dones              = []
            self.next_states        = []
            self.worker_action_datas = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states, self.worker_action_datas = datas

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), np.array(self.worker_action_datas[idx], dtype = np.float32)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states, self.worker_action_datas
    
    def pop(self, idx):
        return self.states.pop(idx), self.actions.pop(idx), self.rewards.pop(idx), self.dones.pop(idx), self.next_states.pop(idx), \
            self.worker_action_datas.pop(idx)
    
    def save_eps(self, state, action, reward, done, next_state, worker_action_data):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.worker_action_datas.append(worker_action_data)     

    def save_replace_all(self, states, actions, rewards, dones, next_states, worker_action_datas):
        self.states                 = states
        self.actions                = actions
        self.rewards                = rewards
        self.dones                  = dones
        self.next_states            = next_states
        self.worker_action_datas    = worker_action_datas
    
    def clear_memory(self, idx = -100):
        if idx == -100:
            del self.states[:]
            del self.actions[:]
            del self.rewards[:]
            del self.dones[:]
            del self.next_states[:]
            del self.worker_action_datas[:]

        else:
            del self.states[idx]
            del self.actions[idx]
            del self.rewards[idx]
            del self.dones[idx]
            del self.next_states[idx]
            del self.worker_action_datas[idx]