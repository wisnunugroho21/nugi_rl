import torch

class QLoss():        

    def __init__(self, distribution, gamma = 0.99, alpha = 0.2):
        self.gamma          = gamma
        self.distribution   = distribution
        self.alpha          = alpha

    def compute_loss(self, predicted_q_value1, predicted_q_value2, target_value, reward, done):
        target_q_value          = (reward + (1 - done) * self.gamma * target_value).detach()

        q_value_loss1           = ((target_q_value - predicted_q_value1).pow(2) * 0.5).mean()
        q_value_loss2           = ((target_q_value - predicted_q_value2).pow(2) * 0.5).mean()
        
        return q_value_loss1 + q_value_loss2