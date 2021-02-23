import torch
from policy_function.advantage_function import AdvantageFunction

class QLoss():
    def __init__(self, value_clip, gamma = 0.99, lam = 0.95):
        self.advantage_function = AdvantageFunction(gamma, lam)
        self.value_clip         = value_clip

    def compute_loss(self, predicted_q_values, old_q_values, rewards, dones, next_values):
        advantages      = self.advantage_function.generalized_advantage_estimation(rewards, predicted_q_values, next_values, dones)
        target_q_value  = (advantages + predicted_q_values).detach()

        if self.value_clip is None:
            q_loss          = ((target_q_value - predicted_q_values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_q_values + torch.clamp(predicted_q_values - old_q_values, -self.value_clip, self.value_clip)
            q_loss          = ((target_q_value - vpredclipped).pow(2) * 0.5).mean()

        return q_loss