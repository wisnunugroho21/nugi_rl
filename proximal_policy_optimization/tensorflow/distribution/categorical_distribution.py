import tensorflow_probability as tfp
import tensorflow as tf

def sample(datas):
    distribution = tfp.distributions.Categorical(probs = datas)
    return distribution.sample()

def entropy(datas):
    distribution = tfp.distributions.Categorical(probs = datas)            
    return distribution.entropy()

def logprob(datas, value_data):
    if len(value_data.shape) > 1:
        value_data = tf.squeeze(value_data)

    distribution = tfp.distributions.Categorical(probs = datas)
    return tf.expand_dims(distribution.log_prob(value_data), 1)

def kl_divergence(datas1, datas2):
    distribution1 = tfp.distributions.Categorical(probs = datas1)
    distribution2 = tfp.distributions.Categorical(probs = datas2)

    return tfp.distributions.kl_divergence(distribution1, distribution2)
    