from torch.distributions import Beta
from torch.distributions.kl import kl_divergence

from distribution.basic_continous import BasicContinous
from helpers.pytorch_utils import set_device, to_list

class BetaContinous(BasicContinous):        
    def sample(self, datas):
        alpha, beta = datas

        distribution    = Beta(alpha, beta)
        action          = distribution.sample()

        return action
        
    def entropy(self, datas):
        alpha, beta = datas

        distribution = Beta(alpha, beta)
        return distribution.entropy()
        
    def logprob(self, datas, value_data):
        alpha, beta = datas

        distribution = Beta(alpha, beta)
        return distribution.log_prob(value_data)

    def kldivergence(self, datas1, datas2):
        alpha1, beta1 = datas1
        alpha2, beta2 = datas2

        distribution1 = Beta(alpha1, beta1)
        distribution2 = Beta(alpha2, beta2)
        return kl_divergence(distribution1, distribution2)