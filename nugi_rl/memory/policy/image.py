import torch

from nugi_rl.memory.policy.standard import PolicyMemory
from nugi_rl.utilities.augmentation.base import Augmentation

class ImagePolicyMemory(PolicyMemory):
    def __init__(self, trans: Augmentation, capacity: int = 100000, datas: tuple = None):
        super().__init__(capacity = capacity, datas = datas)
        
        self.trans = trans

    def __getitem__(self, idx):
        states      = self.trans.augment(self.states[idx])
        next_states = self.trans.augment(self.next_states[idx])
        logprobs    = self.logprobs[idx]
        
        if len(logprobs.shape) == 1:
            logprobs = logprobs.unsqueeze(-1)

        return states, self.actions[idx], self.rewards[idx].unsqueeze(-1), self.dones[idx].unsqueeze(-1), next_states, logprobs