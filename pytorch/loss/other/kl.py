import torch
from distribution.basic import BasicContinous

class KL():
    def __init__(self, distribution):
        self.distribution   = distribution

    def compute_loss(self, mean_pred, std_pred, mean_rand, std_rand):        
        kl  = self.distribution.kldivergence((mean_pred, std_pred), (mean_rand, std_rand)).mean()
        return kl