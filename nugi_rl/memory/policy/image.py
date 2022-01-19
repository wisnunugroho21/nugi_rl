import torch
from torchvision.transforms import Compose

from nugi_rl.memory.policy.standard import PolicyMemory

class ImagePolicyMemory(PolicyMemory):
    def __init__(self, trans: Compose, capacity: int = 100000, datas: tuple = None):
        super().__init__(capacity = capacity, datas = datas)
        
        self.trans = trans

    def __getitem__(self, idx):
        states      = self.trans(self.states[idx])
        next_states = self.trans(self.next_states[idx])
        logprobs    = self.logprobs[idx]
        
        if len(logprobs.shape) == 1:
            logprobs = logprobs.unsqueeze(-1)

        return states, torch.tensor(self.actions[idx]), torch.tensor([self.rewards[idx]]), torch.tensor([self.dones[idx]]), next_states, logprobs