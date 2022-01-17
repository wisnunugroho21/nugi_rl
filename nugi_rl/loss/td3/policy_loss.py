import torch
from torch import Tensor

class PolicyLoss():
    def compute_loss(self, q_value1: Tensor, q_value2: Tensor) -> Tensor:
        policy_loss = -1 * torch.min(q_value1, q_value2).mean()
        return policy_loss