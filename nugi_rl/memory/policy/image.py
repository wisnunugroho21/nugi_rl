import torch

from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.utilities.augmentation.base import Augmentation

class ImagePolicyMemory(PolicyMemory):
    def __init__(self, trans: Augmentation, capacity: int = 100000, datas: tuple = None):
        super().__init__(capacity = capacity, datas = datas)
        
        self.trans = trans

    def __getitem__(self, idx):
        if isinstance(self.states, list):
            states  = []
            for s in self.states:
                states.append(self.trans.augment(s[idx]))

            next_states = []
            for ns in self.next_states:
                next_states.append(self.trans.augment(ns[idx]))
                
        else:
            states      = self.trans.augment(self.states[idx])
            next_states = self.trans.augment(self.next_states[idx])

        return states, self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), next_states, self.logprobs[idx]