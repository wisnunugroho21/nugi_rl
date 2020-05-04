from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch
import torchvision
from utils.pytorch_utils import set_device

def sample(datas):
    distribution = Categorical(datas)
    return distribution.sample()
    
def entropy(datas, use_gpu = True):
    distribution = Categorical(datas)   
    return distribution.entropy().unsqueeze(1)
    
def logprob(datas, value_data, use_gpu = True):
    distribution = Categorical(datas)
    return distribution.log_prob(value_data).unsqueeze(1)

def kldivergence(datas1, datas2, use_gpu = True):
    distribution1 = Categorical(datas1)
    distribution2 = Categorical(datas2)

    return kl_divergence(distribution1, distribution2).unsqueeze(1)
         