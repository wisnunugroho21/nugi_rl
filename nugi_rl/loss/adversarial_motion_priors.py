import torch
import torch.nn as nn
from torch import Tensor

class DiscriminatorLoss(nn.Module):
    def __init__(self, coef: int = 10) -> None:
        self.coef = coef

    def forward(self, dis_expert: Tensor, dis_policy: Tensor, policy_states: Tensor, policy_next_states: Tensor, goals: Tensor) -> Tensor:
        gradient_norm       = torch.autograd.grad(dis_policy, [policy_states, policy_next_states, goals]).square().sum(-1).sqrt()

        expert_loss         = (dis_expert - 1).pow(2).mean()
        policy_loss         = (dis_policy + 1).pow(2).mean()
        gradient_penalty    = self.coef / 2 * gradient_norm.mean()

        return expert_loss + policy_loss + gradient_penalty