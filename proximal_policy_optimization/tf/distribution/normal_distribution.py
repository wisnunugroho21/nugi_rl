import tensorflow as tf
from tensorflow_probability.python.distributions import Normal, kl_divergence

def sample(mean, std):
    distribution = Normal(mean, std)
    return distribution.sample()
    
def entropy(mean, std):
    distribution = Normal(mean, std) 
    return distribution.entropy()
    
def logprob(mean, std, value_data):
    distribution = Normal(mean, std)
    return distribution.log_prob(value_data)

def kldivergence(mean1, std1, mean2, std2):
    distribution1 = Normal(mean1, std1)
    distribution2 = Normal(mean2, std2)

    return kl_divergence(distribution1, distribution2)