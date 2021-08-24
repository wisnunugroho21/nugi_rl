from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch

from helpers.pytorch_utils import set_device, to_list

class BasicDiscrete():
    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().int()
        
    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().unsqueeze(1)
        
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)        
        return distribution.log_prob(value_data).unsqueeze(1)

    def kldivergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)
        return kl_divergence(distribution1, distribution2).unsqueeze(1)

    def deterministic(self, datas):
        return int(torch.argmax(datas, 1))
