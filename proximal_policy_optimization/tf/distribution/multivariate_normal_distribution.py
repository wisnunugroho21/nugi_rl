import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag, kl_divergence

def sample(mean, std):
    distribution = MultivariateNormalDiag(mean, std)
    return distribution.sample()
    
def entropy(mean, std):
    distribution = MultivariateNormalDiag(mean, std) 
    return distribution.entropy()
    
def logprob(mean, std, value_data):
    distribution = MultivariateNormalDiag(mean, std)
    return tf.expand_dims(distribution.log_prob(value_data), 1)

def kldivergence(mean1, std1, mean2, std2):
    distribution1 = MultivariateNormalDiag(mean1, std1)
    distribution2 = MultivariateNormalDiag(mean2, std2)

    return kl_divergence(distribution1, distribution2)