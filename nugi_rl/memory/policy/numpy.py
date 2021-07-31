import numpy as np
import torch

from memory.policy.standard import PolicyMemory

class NumpyPolicyMemory(PolicyMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.states         = np.array([], dtype = np.float32)
            self.actions        = np.array([], dtype = np.float32)
            self.rewards        = np.array([], dtype = np.float32)
            self.dones          = np.array([], dtype = np.float32)
            self.next_states    = np.array([], dtype = np.float32)
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas

    def __len__(self):
        return self.dones.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx]), torch.from_numpy(self.rewards[idx]), \
            torch.from_numpy(self.dones[idx]), torch.from_numpy(self.next_states[idx])     

    def save_obs(self, state, action, reward, done, next_state):
        if len(self) == 0:
            self.states         = np.array([state], dtype = np.float32)
            self.actions        = np.array([action], dtype = np.float32)
            self.rewards        = np.array([[reward]], dtype = np.float32)
            self.dones          = np.array([[done]], dtype = np.float32)
            self.next_states    = np.array([next_state], dtype = np.float32)

        else:
            self.states         = np.append(self.states, [state], axis = 0)
            self.actions        = np.append(self.actions, [action], axis = 0)
            self.rewards        = np.append(self.rewards, [[reward]], axis = 0)
            self.dones          = np.append(self.dones, [[done]], axis = 0)
            self.next_states    = np.append(self.next_states, [next_state], axis = 0)

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()

        self.states         = np.array(states)
        self.actions        = np.array(actions)
        self.rewards        = np.array(rewards)
        self.dones          = np.array(dones)
        self.next_states    = np.array(next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        self.states         = np.concatenate(self.states, np.array(states))
        self.actions        = np.concatenate(self.actions, np.array(actions))
        self.rewards        = np.concatenate(self.rewards, np.array(rewards))
        self.dones          = np.concatenate(self.dones, np.array(dones))
        self.next_states    = np.concatenate(self.next_states, np.array(next_states))

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None or -1:
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
        self.states         = np.delete(self.states, np.s_[:])
        self.actions        = np.delete(self.actions, np.s_[:])
        self.rewards        = np.delete(self.rewards, np.s_[:])
        self.dones          = np.delete(self.dones, np.s_[:])
        self.next_states    = np.delete(self.next_states, np.s_[:])
