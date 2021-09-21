import torch

class AlphaLoss():
    def __init__(self, distribution, coef_alpha_mean_upper = torch.Tensor([0.01]), coef_alpha_mean_below = torch.Tensor([0.005]), 
            coef_alpha_cov_upper = torch.Tensor([0.01]), coef_alpha_cov_below = torch.Tensor([0.005])):
        self.distribution           = distribution

        self.coef_alpha_mean_upper  = coef_alpha_mean_upper
        self.coef_alpha_mean_below  = coef_alpha_mean_below
        self.coef_alpha_cov_upper   = coef_alpha_cov_upper
        self.coef_alpha_cov_below   = coef_alpha_cov_below

    def compute_loss(self, action_datas, old_action_datas, alpha):
        alpha_mean, alpha_cov   = alpha

        coef_mean_alpha         = torch.distributions.Uniform(self.coef_alpha_mean_below.log(), self.coef_alpha_mean_upper.log()).sample().exp()
        coef_cov_alpha          = torch.distributions.Uniform(self.coef_alpha_cov_below.log(), self.coef_alpha_cov_upper.log()).sample().exp()

        Kl_mean                 = self.distribution.kldivergence_mean(old_action_datas, action_datas)
        Kl_cov                  = self.distribution.kldivergence_cov(old_action_datas, action_datas)
       
        mean_loss               = alpha_mean * (coef_mean_alpha - Kl_mean.squeeze().detach()) + alpha_mean.detach() * Kl_mean.squeeze()
        cov_loss                = alpha_cov * (coef_cov_alpha - Kl_cov.squeeze().detach()) + alpha_cov.detach() * Kl_cov.squeeze()

        return mean_loss.mean() + cov_loss.mean()