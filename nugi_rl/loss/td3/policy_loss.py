import torch
import torch.nn as nn
from torch import Tensor

class PolicyLoss(nn.Module):
    def forward(self, q_value1: Tensor, q_value2: Tensor) -> Tensor:
        policy_loss = -1 * torch.min(q_value1, q_value2).mean()
        return policy_loss