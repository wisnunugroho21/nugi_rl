import torch
import torch.nn as nn
from torch import Tensor

class QLoss(nn.Module):
    def __init__(self, gamma = 0.99):
        super().__init__()
        self.gamma  = gamma

    def forward(self, predicted_q1: Tensor, predicted_q2: Tensor, target_next_q1: Tensor, target_next_q2: Tensor, reward: Tensor, done: Tensor) -> Tensor:
        target_value            = torch.min(target_next_q1, target_next_q2)
        target_q_value          = (reward + (1 - done) * self.gamma * target_value).detach()

        q_value_loss1           = ((target_q_value - predicted_q1).pow(2) * 0.5).mean()
        q_value_loss2           = ((target_q_value - predicted_q2).pow(2) * 0.5).mean()
        
        return q_value_loss1 + q_value_loss2