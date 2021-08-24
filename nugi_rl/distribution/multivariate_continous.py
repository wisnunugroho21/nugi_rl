from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from distribution.basic_continous import BasicContinous
from helpers.pytorch_utils import set_device, to_list

class MultivariateContinous(BasicContinous):
    def sample(self, datas):
        mean, std = datas

        distribution    = MultivariateNormal(mean, std)
        action          = distribution.sample().squeeze(0)
        return action
        
    def entropy(self, datas):
        mean, std = datas

        distribution = MultivariateNormal(mean, std) 
        return distribution.entropy()
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = MultivariateNormal(mean, std)
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = MultivariateNormal(mean1, std1)
        distribution2 = MultivariateNormal(mean2, std2)
        return kl_divergence(distribution1, distribution2)

    def act_deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)