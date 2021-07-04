import torch

class PolicyLoss():
    def __init__(self, distribution, alpha = 0.2):
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, action_datas, actions, q_value1, q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        policy_loss             = (self.alpha * log_prob - torch.min(q_value1, q_value2)).mean()
        return policy_loss