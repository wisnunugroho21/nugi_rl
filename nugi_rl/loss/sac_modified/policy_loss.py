import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.loss.sac.policy_loss import PolicyLoss


class ModifiedPolicyLoss(PolicyLoss):
    def __init__(self, distribution: Distribution):
        super().__init__(distribution)

    def forward(
        self,
        action_datas: Tensor,
        actions: Tensor,
        q_value1: Tensor,
        q_value2: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        log_prob = self.distribution.logprob(action_datas, actions)
        policy_loss = (alpha * log_prob - torch.min(q_value1, q_value2)).mean()
        return policy_loss
