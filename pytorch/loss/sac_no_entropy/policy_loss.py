import torch

class PolicyLoss():
    def compute_loss(self, q_value):
        policy_loss = q_value.mean() * -1
        return policy_loss