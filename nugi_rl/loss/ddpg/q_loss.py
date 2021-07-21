import torch

class QLoss():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, predicted_q_value, target_next_q, reward, done):
        target_q_value  = (reward + (1 - done) * self.gamma * target_next_q).detach()
        q_value_loss    = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()    
        return q_value_loss