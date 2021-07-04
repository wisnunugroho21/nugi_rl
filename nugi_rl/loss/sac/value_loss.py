import torch

class ValueLoss():
    def __init__(self, distribution, alpha = 0.2):
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, predicted_value, action_datas, actions, q_value1, q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        target_value            = (torch.min(q_value1, q_value2) - self.alpha * log_prob).detach()

        value_loss              = ((target_value - predicted_value).pow(2) * 0.5).mean()
        return value_loss