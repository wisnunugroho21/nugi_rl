import torch
from torch import Tensor

class MonteCarloDiscounted():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_value(self, reward: Tensor, done: Tensor) -> Tensor:
        returns = []        
        running_add = 0
        
        for i in reversed(range(len(reward))):
            running_add = reward[i] + (1.0 - done) * self.gamma * running_add  
            returns.insert(0, running_add)
            
        return torch.stack(returns)