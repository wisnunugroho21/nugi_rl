import torch.nn as nn
from torch import Tensor

class PolicyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, q_value: Tensor) -> Tensor:
        policy_loss = q_value.mean() * -1
        return policy_loss