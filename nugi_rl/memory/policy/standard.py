from copy import deepcopy
import torch
from torch.utils.data import Dataset

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity
        self.position       = 0

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx]), torch.FloatTensor(self.actions[idx]), torch.FloatTensor([self.rewards[idx]]), \
            torch.FloatTensor([self.dones[idx]]), torch.FloatTensor(self.next_states[idx])

    def save_obs(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(deepcopy(state))
        self.actions.append(deepcopy(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(deepcopy(next_state))

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, rewards, dones, next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_obs(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or end_position == -1:
            states      = self.states[start_position:end_position + 1]
            actions     = self.actions[start_position:end_position + 1]
            rewards     = self.rewards[start_position:end_position + 1]
            dones       = self.dones[start_position:end_position + 1]
            next_states = self.next_states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]
            actions     = self.actions[start_position:]
            rewards     = self.rewards[start_position:]
            dones       = self.dones[start_position:]
            next_states = self.next_states[start_position:]

        return states, actions, rewards, dones, next_states 

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

    def clear_idx(self, idx):
        del self.states[idx]
        del self.actions[idx]
        del self.rewards[idx]
        del self.dones[idx]
        del self.next_states[idx]