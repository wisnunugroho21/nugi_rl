from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch

from distribution.basic_continous import BasicContinous
from helpers.pytorch_utils import set_device, to_list

class TanhClippedContinous(BasicContinous):
    def __init__(self, use_gpu):
        self.use_gpu    = use_gpu
        
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        old_logprob = distribution.log_prob(value_data).float().to(set_device(self.use_gpu))

        return old_logprob - (1.0 - value_data.tanh().pow(2)).log()