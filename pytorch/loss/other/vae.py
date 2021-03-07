import torch

class CLR():
    def __init__(self, distribution):
        self.distribution   = distribution

    def compute_loss(self, states, reconstruc_states, mean_pred, std_pred, mean_rand, std_rand):        
        kl          = self.distribution.kldivergence((mean_pred, std_pred), (mean_rand, std_rand)).mean()
        dif_states  = ((reconstruc_states - states).pow(2) * 0.5).mean()

        return dif_states + kl