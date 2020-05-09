import tensorflow as tf
from utils.pytorch_utils import set_device

def impala_advantage_estimation(rewards, values, NextReturns, dones, worker_logprobs, learner_logprobs, gamma = 0.99, lam = 0.95):
    po          = tf.math.reduce_mean(tf.math.minimum(1.0, tf.math.exp(learner_logprobs - worker_logprobs)), 1)
    return po * (rewards + (1.0 - dones) * gamma * NextReturns - values)