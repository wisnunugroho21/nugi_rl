import torch
from torch import Tensor
from nugi_rl.distribution.discrete.basic_discrete import BasicDiscrete
from nugi_rl.loss.v_mpo.alpha.base import AlphaLoss

class DiscreteAlphaLoss(AlphaLoss):
    def __init__(self, distribution: BasicDiscrete, coef_alpha_upper: Tensor = torch.Tensor([0.01]), coef_alpha_below: Tensor = torch.Tensor([0.005])):
        self.distribution       = distribution

        self.coef_alpha_upper  = coef_alpha_upper
        self.coef_alpha_below  = coef_alpha_below

    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, alpha: tuple) -> Tensor:
        alpha       = alpha[0]
        
        coef_alpha  = torch.distributions.Uniform(self.coef_alpha_below.log(), self.coef_alpha_upper.log()).sample().exp()
        Kl          = self.distribution.kldivergence(old_action_datas, action_datas)
       
        loss        = alpha * (coef_alpha - Kl.squeeze().detach()) + alpha.detach() * Kl.squeeze()
        return loss.mean()