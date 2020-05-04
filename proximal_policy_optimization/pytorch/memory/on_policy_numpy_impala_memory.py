import torch
import numpy as np
from torch.utils.data import Dataset

class OnMemory(Dataset):
    def __init__(self, datas = None):
        if datas is None :
            self.states             = np.array([])
            self.actions            = np.array([])
            self.rewards            = np.array([])
            self.dones              = np.array([])
            self.next_states        = np.array([])
            self.logprobs           = np.array([])
            self.next_next_states   = np.array([])
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs, self.next_next_states = datas

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx], self.next_states[idx], self.logprobs[idx], self.next_next_states[idx]      

    def save_eps(self, state, action, reward, done, next_state, logprob, next_next_state = None):
        if self.__len__() == 0:
            self.states             = np.array([state])
            self.actions            = np.array([action])
            self.rewards            = np.array([[reward]])
            self.dones              = np.array([[done]])
            self.next_states        = np.array([next_state])
            self.logprobs           = np.array([logprob])

            if next_next_state is not None:
                self.next_next_states   = np.array([next_next_state])

        else:
            self.states             = np.append(self.states, [state], axis = 0)
            self.actions            = np.append(self.actions, [action], axis = 0)
            self.rewards            = np.append(self.rewards, [[reward]], axis = 0)
            self.dones              = np.append(self.dones, [[done]], axis = 0)
            self.next_states        = np.append(self.next_states, [next_state], axis = 0)
            self.logprobs           = np.append(self.logprobs, [logprob], axis = 0)

            if next_next_state is not None:
                self.next_next_states   = np.append(self.next_next_states, [next_next_state], axis = 0)            

    def save_replace_all(self, states, actions, rewards, dones, next_states, logprobs, next_next_states):
        self.states             = np.array(states)
        self.actions            = np.array(actions)
        self.rewards            = np.array(rewards)
        self.dones              = np.array(dones)
        self.next_states        = np.array(next_states)
        self.logprobs           = np.array(logprobs)

        if next_next_states is not None:
            self.next_next_states   = np.array(next_next_states)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states, self.logprobs, self.next_next_states

    def clearMemory(self):
        self.states             = np.delete(self.states, np.s_[:])
        self.actions            = np.delete(self.actions, np.s_[:])
        self.rewards            = np.delete(self.rewards, np.s_[:])
        self.dones              = np.delete(self.dones, np.s_[:])
        self.next_states        = np.delete(self.next_states, np.s_[:])
        self.logprobs           = np.delete(self.logprobs, np.s_[:])
        self.next_next_states   = np.delete(self.next_next_states, np.s_[:])

    def convert_next_states_to_next_next_states(self):
        self.next_next_states       = np.roll(self.next_states, -1, axis = 0)
        self.next_next_states[-1]   = self.next_next_states[-2]

