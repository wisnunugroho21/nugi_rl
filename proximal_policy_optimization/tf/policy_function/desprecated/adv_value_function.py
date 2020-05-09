import tensorflow as tf

def generalized_advantage_value_estimation_impala(rewards, values, next_values, dones, worker_logprobs, learner_logprobs, next_next_states, gamma = 0.99, lam = 0.95):
    running_add = 0
    v_traces    = []
    adv         = []
    tres        = tf.fill(worker_logprobs.shape, 1.0)

    co          = tf.math.reduce_mean(tf.math.minimum(tres, tf.math.exp(learner_logprobs - worker_logprobs)), 1)
    po          = tf.math.reduce_mean(tf.math.minimum(tres, tf.math.exp(learner_logprobs - worker_logprobs)), 1)
    delta       = po * (rewards + (1.0 - dones) * gamma * next_values - values)

    for i in reversed(range(len(values))):        
        running_add = values[i] + delta[i] + (1.0 - dones) * gamma * co[i] * (running_add - next_values[i])
        v_traces.insert(0, running_add)        
               
    v_traces    = tf.stack(v_traces)
    adv         = tf.stack(adv)
    return adv, v_traces