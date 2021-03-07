from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
import torch

from utils.pytorch_utils import set_device, to_numpy

class BasicDiscrete():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().int().to(set_device(self.use_gpu))
        
    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().unsqueeze(1).float().to(set_device(self.use_gpu))
        
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)        
        return distribution.log_prob(value_data).unsqueeze(1).float().to(set_device(self.use_gpu))

    def kldivergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)
        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(set_device(self.use_gpu))

    def deterministic(self, datas):
        return int(torch.argmax(datas, 1))

class BasicContinous():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def sample(self, datas):
        mean, std = datas

        distribution = Normal(mean, std)
        action       = distribution.sample().squeeze(0).float().to(set_device(self.use_gpu))
        return action
        
    def entropy(self, datas):
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(set_device(self.use_gpu))
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(set_device(self.use_gpu))

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2).float().to(set_device(self.use_gpu))

    def deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)