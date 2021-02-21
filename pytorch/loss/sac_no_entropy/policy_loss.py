import torch

class PolicyLoss():
    def compute_loss(self, q_value1, q_value2):
        q_value     = torch.min(q_value1, q_value2)
        policy_loss = q_value.mean() * -1
        return policy_loss