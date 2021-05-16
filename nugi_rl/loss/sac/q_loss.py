import torch

class QLoss():        

    def __init__(self, distribution, gamma = 0.99):
        self.gamma          = gamma
        self.distribution   = distribution

    def compute_loss(self, predicted_q_value, target_q_value1, target_q_value2, next_action_datas, next_actions, reward, done):
        next_log_prob           = self.distribution.logprob(next_action_datas, next_actions)
        next_value              = (torch.min(target_q_value1, target_q_value2) - next_log_prob).detach()

        target_q_value          = (reward + (1 - done) * self.gamma * next_value).detach()

        q_value_loss            = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()
        return q_value_loss