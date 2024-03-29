import torch
import torch.nn as nn
from torch import Tensor

class GeneralizedAdvantageEstimation(nn.Module):
    def __init__(self, gamma = 0.99):
        super().__init__()
        
        self.gamma  = gamma

    def forward(self, rewards: Tensor, values: Tensor, next_values: Tensor, dones: Tensor) -> Tensor:
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            adv.insert(0, gae)

        adv = torch.stack(adv)
        return adv - values