import torch

class AlphaLoss():
    def __init__(self, distribution, coef_alpha_upper = torch.Tensor([0.01]), coef_alpha_below = torch.Tensor([0.005])):
        self.distribution       = distribution

        self.coef_alpha_upper  = coef_alpha_upper
        self.coef_alpha_below  = coef_alpha_below

    def compute_loss(self, action_datas, old_action_datas, alpha):
        coef_alpha  = torch.distributions.Uniform(self.coef_alpha_below.log(), self.coef_alpha_upper.log()).sample().exp()
        Kl          = self.distribution.kldivergence(old_action_datas, action_datas)
       
        loss        = alpha * (coef_alpha - Kl.squeeze().detach()) + alpha.detach() * Kl.squeeze()
        return loss.mean()