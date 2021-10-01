import torch
import math

class EntropyLoss():
    def __init__(self, distribution, entropy_coef = 0.1):
        self.distribution       = distribution
        self.entropy_coef       = entropy_coef

    def compute_loss(self, action_datas):
        loss                = -1 * 0.1 * self.distribution.entropy(action_datas).mean()

        return loss
