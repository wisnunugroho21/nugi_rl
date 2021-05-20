import torch

class PolicyLoss():
    def __init__(self, distribution, alpha = 0.2):
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, action_datas, actions, predicted_q_value1, predicted_q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        policy_loss             = (torch.min(predicted_q_value1, predicted_q_value2) - self.alpha * log_prob).mean()
        return policy_loss * -1