import torch

class QLoss():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, q1_value, q2_value, naive_q1_value, naive_q2_value, target_next_value, rewards, dones):
        target_q_value          = (rewards + (1 - dones) * self.gamma * target_next_value).detach()

        td_error1               = ((target_q_value - q1_value).pow(2) * 0.5).mean()
        td_error2               = ((target_q_value - q2_value).pow(2) * 0.5).mean()

        cql_regularizer1        = (naive_q1_value - q1_value).mean()
        cql_regularizer2        = (naive_q2_value - q2_value).mean()

        q1_value_loss           = td_error1 + cql_regularizer1
        q2_value_loss           = td_error2 + cql_regularizer2

        return q1_value_loss + q2_value_loss