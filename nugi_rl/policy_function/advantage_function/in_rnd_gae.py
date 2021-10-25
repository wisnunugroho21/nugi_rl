import torch
from torch import Tensor

from nugi_rl.policy_function.advantage_function.gae import GeneralizedAdvantageEstimation

class InRndGeneralizedAdvantageEstimation(GeneralizedAdvantageEstimation):
    def __init__(self, gamma= 0.99):
        super().__init__(gamma = gamma)

    def compute_advantages(self, state_targets: Tensor, state_preds: Tensor, std_in_rewards: Tensor, 
        values: Tensor, next_values: Tensor, dones: Tensor) -> Tensor:
        
        rewards = ((state_targets - state_preds).pow(2) * 0.5 / (std_in_rewards.mean() + 1e-6)).detach()
        return super().compute_advantages(rewards, values, next_values, dones)