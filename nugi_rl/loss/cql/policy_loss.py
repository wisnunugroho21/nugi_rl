import torch

class OffPolicyLoss():
    def compute_loss(self, q1_value, q2_value):
        policy_loss = (torch.min(q1_value, q2_value) * -1).mean()
        return policy_loss 