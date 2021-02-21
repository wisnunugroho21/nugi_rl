import torch

class PolicyLoss():
    def compute_loss(self, predicted_new_q_value1, predicted_q_value2, log_prob):
        predicted_new_q_value   = torch.min(predicted_new_q_value1, predicted_q_value2)
        policy_loss             = (predicted_new_q_value - log_prob).mean()

        return policy_loss