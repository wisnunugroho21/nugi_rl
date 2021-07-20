import torch

class OffPolicyLoss():
    def compute_loss(self, predicted_q_value):
        policy_loss = (predicted_q_value * -1).mean() 
        return policy_loss