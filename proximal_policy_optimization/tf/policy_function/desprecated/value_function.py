import tensorflow as tf
from utils.pytorch_utils import set_device

def vtrace(rewards, values, next_values, dones, worker_logprobs, learner_logprobs, gamma = 0.99, lam = 0.95):
    running_add = 0
    v_traces    = []

    co          = tf.math.reduce_mean(tf.math.minimum(tres, tf.math.exp(learner_logprobs - worker_logprobs)), 1)
    po          = tf.math.reduce_mean(tf.math.minimum(tres, tf.math.exp(learner_logprobs - worker_logprobs)), 1)

    delta       = po * (rewards + (1.0 - dones) * gamma * next_values - values)
    for i in reversed(range(len(values))):        
        #running_add = values[i] + delta[i] + gamma * co[i] * (running_add - next_values[i])
        #running_add = delta[i] + gamma * co[i] * (running_add - next_values[i])
        running_add = delta[i] + (1.0 - dones[i]) * gamma * co[i] * running_add
        v_traces.insert(0, running_add)        
               
    v_traces    = tf.stack(v_traces)
    return v_traces + values