import tensorflow_probability as tfp
import tensorflow as tf

def sample(mean, std):
    distribution = tfp.distributions.MultivariateNormalDiag(mean, std)
    return distribution.sample()

def entropy(mean, std):
    distribution = tfp.distributions.MultivariateNormalDiag(mean, std)            
    return distribution.entropy()

def logprob(mean, std, value_data):
    if len(value_data) == 1:
        value_data = tf.expand_dims(value_data, 1)

    distribution = tfp.distributions.MultivariateNormalDiag(mean, std)
    return tf.expand_dims(distribution.log_prob(value_data), 1)

def kl_divergence(mean1, std1, mean2, std2):
    distribution1 = tfp.distributions.MultivariateNormalDiag(mean1, std1)
    distribution2 = tfp.distributions.MultivariateNormalDiag(mean2, std2)

    return tfp.distributions.kl_divergence(distribution1, distribution2)
    