import torch
from torch import Tensor

class GeneralizedAdvantageEstimation():
    def __init__(self, gamma = 0.99):
        self.gamma  = gamma

    def compute_advantages(self, rewards: Tensor, values: Tensor, next_values: Tensor, dones: Tensor) -> Tensor:
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)