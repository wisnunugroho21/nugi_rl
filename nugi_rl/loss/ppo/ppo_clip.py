import torch
from torch import Tensor

from nugi_rl.distribution.base import Distribution
from nugi_rl.policy_function.advantage_function.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from nugi_rl.loss.ppo.base import PPO

class PPOClip(PPO):
    def __init__(self, distribution: Distribution, advantage_function: GeneralizedAdvantageEstimation, 
        policy_clip: float = 0.2, value_clip: float = 1.0, vf_loss_coef: float = 1.0, entropy_coef: float = 0.01):

        self.policy_clip        = policy_clip
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.advantage_function = advantage_function
        self.distribution       = distribution
 
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, values: Tensor, old_values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        advantages      = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        returns         = (advantages + values).detach()

        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp() 
        surr1           = ratios * advantages
        surr2           = ratios.clamp(1 - self.policy_clip, 1 + self.policy_clip) * advantages
        pg_loss         = torch.min(surr1, surr2).mean()

        dist_entropy    = self.distribution.entropy(action_datas).mean()

        if self.value_clip is None:
            critic_loss     = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            critic_loss     = ((returns - vpredclipped).pow(2) * 0.5).mean()

        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss