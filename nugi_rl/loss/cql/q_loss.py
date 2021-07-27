import torch

class QLoss():
    def __init__(self, gamma = 0.99, alpha = 1.0):
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, q1_value, q2_value, naive_q1_value, naive_q2_value, target_next_q1, target_next_q2, rewards, dones):
        target_q_value          = (rewards + (1 - dones) * self.gamma * torch.min(target_next_q1, target_next_q2)).detach()

        td_error1               = ((target_q_value - q1_value).pow(2) * 0.5).mean()
        td_error2               = ((target_q_value - q2_value).pow(2) * 0.5).mean()

        cql_regularizer1        = ((naive_q1_value - q1_value) * self.alpha).mean()
        cql_regularizer2        = ((naive_q2_value - q2_value) * self.alpha).mean()

        q1_value_loss           = td_error1 + cql_regularizer1
        q2_value_loss           = td_error2 + cql_regularizer2

        return q1_value_loss + q2_value_loss