import torch
from policy_function.advantage_function import AdvantageFunction

class GeneralizedValue():
    def __init__(self, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01, gamma = 0.99):
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.advantage_function = AdvantageFunction(gamma)

    def compute_loss(self, values, old_values, next_values, rewards, dones):
        advantages      = self.advantage_function.generalized_advantage_estimation(rewards, values, next_values, dones)
        returns         = (advantages + values).detach()

        if self.value_clip is None:
            critic_loss     = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            critic_loss     = ((returns - vpredclipped).pow(2) * 0.5).mean()

        loss = critic_loss * self.vf_loss_coef
        return loss