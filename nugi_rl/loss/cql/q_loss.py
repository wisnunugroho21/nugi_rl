import torch

class QLoss():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, q1_value, q2_value, naive_q1_value, naive_q2_value, target_next_value, rewards, dones):
        target_q_value      = (rewards + (1 - dones) * self.gamma * target_next_value).detach()
        cql_regularizer     = torch.min(naive_q1_value, naive_q2_value) - torch.min(q1_value, q2_value).mean()

        q1_value_loss       = ((target_q_value - q1_value).pow(2) * 0.5).mean()
        q2_value_loss       = ((target_q_value - q2_value).pow(2) * 0.5).mean()

        return q1_value_loss + q2_value_loss + cql_regularizer