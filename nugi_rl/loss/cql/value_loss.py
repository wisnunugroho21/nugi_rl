import torch

class ValueLoss():
    def compute_loss(self, predicted_value, q1_value, q2_value):
        value_loss  = ((torch.min(q1_value, q2_value) - predicted_value).pow(2) * 0.5).mean()
        return value_loss