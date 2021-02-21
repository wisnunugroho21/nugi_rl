import torch

class VLoss():
    def compute_loss(self, predicted_new_q_value1, predicted_new_q_value2, log_prob, predicted_value):
        predicted_new_q_value   = torch.min(predicted_new_q_value1, predicted_new_q_value2)
        target_value_func       = predicted_new_q_value - log_prob
        target_value_func       = target_value_func.detach()

        value_loss              = ((target_value_func - predicted_value).pow(2) * 0.5).mean()
        return value_loss