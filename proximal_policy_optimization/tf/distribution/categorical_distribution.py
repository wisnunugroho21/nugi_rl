import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical, kl_divergence

def sample(datas):
    distribution = Categorical(probs = datas)
    return distribution.sample()
    
def entropy(datas):
    distribution = Categorical(probs = datas)   
    return distribution.entropy()
    
def logprob(datas, value_data):
    distribution = Categorical(probs = datas)
    return tf.expand_dims(distribution.log_prob(value_data), 1)

def kldivergence(datas1, datas2):
    distribution1 = Categorical(probs = datas1)
    distribution2 = Categorical(probs = datas2)

    return kl_divergence(distribution1, distribution2)
         