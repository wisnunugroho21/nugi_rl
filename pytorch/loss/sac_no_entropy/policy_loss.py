import torch

class PolicyLoss():
    def compute_loss(self, predicted_q_value1, predicted_q_value2):
        q_value     = torch.min(predicted_q_value1, predicted_q_value2)
        policy_loss = q_value.mean()
        return policy_loss * -1