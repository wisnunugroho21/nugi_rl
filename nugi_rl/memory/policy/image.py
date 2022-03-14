import torch

from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.utilities.augmentation.base import Augmentation

class ImagePolicyMemory(PolicyMemory):
    def __init__(self, trans: Augmentation, capacity: int = 100000, datas: tuple = None):
        super().__init__(capacity = capacity, datas = datas)
        
        self.trans = trans

    def __getitem__(self, idx):
        if isinstance(self.states, list):
            states  = [self.trans(s[idx]) for s in self.states]
            next_states = [self.trans(ns[idx]) for ns in self.next_states]
                
        else:
            states      = self.trans(self.states[idx])
            next_states = self.trans(self.next_states[idx])

        return states, self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), next_states, self.logprobs[idx]