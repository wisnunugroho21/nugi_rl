import torch

class OffPolicyLoss():
    def compute_loss(self, predicted_q_value1):
        policy_loss = (predicted_q_value1 * -1).mean()
        return policy_loss 