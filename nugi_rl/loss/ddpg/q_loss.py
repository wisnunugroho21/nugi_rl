import torch
import torch.nn as nn
from torch import Tensor

class QLoss(nn.Module):
    def __init__(self, gamma: int = 0.99):
        super().__init__()
        
        self.gamma  = gamma

    def forward(self, predicted_q_value: Tensor, target_next_q: Tensor, reward: Tensor, done: Tensor) -> Tensor:
        target_q_value  = (reward + (1 - done) * self.gamma * target_next_q).detach()
        q_value_loss    = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()
            
        return q_value_loss