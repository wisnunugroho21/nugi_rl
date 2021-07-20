import torch

class QLoss():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, predicted_q_value1, naive_predicted_q_value1, predicted_q_value2, naive_predicted_q_value2, target_next_q1, target_next_q2, reward, done):
        # naive_predicted_q_value = torch.min(naive_predicted_q_value1, naive_predicted_q_value2)
        # predicted_q_value       = torch.min(predicted_q_value1, predicted_q_value2)
        # cql_regularizer         = (naive_predicted_q_value - predicted_q_value).mean()

        cql_regularizer1        = (naive_predicted_q_value1 - predicted_q_value1).mean()
        cql_regularizer2        = (naive_predicted_q_value2 - predicted_q_value2).mean()
        
        next_value              = torch.min(target_next_q1, target_next_q2)
        target_q_value          = (reward + (1 - done) * self.gamma * next_value).detach()

        # q_value_loss1           = ((target_q_value - predicted_q_value1).pow(2) * 0.5).mean()
        # q_value_loss2           = ((target_q_value - predicted_q_value2).pow(2) * 0.5).mean()

        td1                     = ((target_q_value - predicted_q_value1).pow(2) * 0.5).mean()
        td2                     = ((target_q_value - predicted_q_value2).pow(2) * 0.5).mean()

        q_value_loss1           = td1 + cql_regularizer1
        q_value_loss2           = td2 + cql_regularizer2
        
        return q_value_loss1 + q_value_loss2