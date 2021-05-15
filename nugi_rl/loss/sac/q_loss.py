import torch

class QLoss():        

    def __init__(self, distribution, gamma = 0.99):
        self.gamma          = gamma
        self.distribution   = distribution

    def compute_loss(self, predicted_q_value, target_q_value1, target_q_value2, action_datas, action, reward, done):
        log_prob                = self.distribution.logprob(action_datas, action)
        next_value              = (torch.min(target_q_value1, target_q_value2) - log_prob).detach()

        target_q_value          = reward + (1 - done) * self.gamma * next_value
        target_q_value          = target_q_value.detach()

        q_value_loss            = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()
        return q_value_loss