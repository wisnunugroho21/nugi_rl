import tensorflow_probability as tfp
import tensorflow as tf

def sample(datas):
    distribution = tfp.distributions.Poisson(rate = datas)
    return distribution.sample()

def entropy(datas):
    distribution = tfp.distributions.Poisson(rate = datas)            
    return distribution.entropy()

def logprob(datas, value_data):
    if len(value_data) == 1:
        value_data = tf.expand_dims(value_data, 1)
        
    distribution = tfp.distributions.Poisson(rate = datas)
    return distribution.log_prob(value_data)

def kl_divergence(datas1, datas2):
    distribution1 = tfp.distributions.Poisson(rate = datas1)
    distribution2 = tfp.distributions.Poisson(rate = datas2)

    return tfp.distributions.kl_divergence(distribution1, distribution2)
    