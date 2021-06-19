import torch

class QLoss():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, predicted_q_value1, predicted_q_value2, target_next_q1, target_next_q2, rewards, dones):
        next_value              = torch.min(target_next_q1, target_next_q2).detach()
        target_q_value          = (rewards + (1 - dones) * self.gamma * next_value).detach()

        td_error1               = (target_q_value - predicted_q_value1).pow(2) * 0.5
        td_error2               = (target_q_value - predicted_q_value2).pow(2) * 0.5

        return (td_error1 + td_error2).mean()