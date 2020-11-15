from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
import torch
import torchvision

class BasicDiscrete():
    def __init__(self, device):
        self.device = device

    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(self.device)
        
    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().unsqueeze(1).float().to(self.device)
        
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(self.device)

    def kldivergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(self.device)

class BasicContinous():
    def __init__(self, device):
        self.device = device

    def sample(self, mean, std):
        distribution = Normal(mean, std)
        return distribution.sample().float().to(self.device)
        
    def entropy(self, mean, std):
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(self.device)
        
    def logprob(self, mean, std, value_data):
        distribution = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(self.device)

    def kldivergence(self, mean1, std1, mean2, std2):
        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(self.device)