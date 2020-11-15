from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch
import torchvision

from distribution.basic_continous import BasicContinous

class MultivariateContinous(BasicContinous):
    def sample(self, mean, std):
        distribution = MultivariateNormal(mean, std)
        return distribution.sample().float().to(self.device)
        
    def entropy(self, mean, std):
        distribution = MultivariateNormal(mean, std) 
        return distribution.entropy().unsqueeze(1).float().to(self.device)
        
    def logprob(self, mean, std, value_data):
        distribution = MultivariateNormal(mean, std)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(self.device)

    def kldivergence(self, mean1, std1, mean2, std2):
        distribution1 = MultivariateNormal(mean1, std1)
        distribution2 = MultivariateNormal(mean2, std2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(self.device)