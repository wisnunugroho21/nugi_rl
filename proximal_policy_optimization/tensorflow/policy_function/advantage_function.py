import tensorflow as tf

def generalized_advantage_estimation(values, rewards, next_value, done, gamma = 0.99, lam = 0.95):
    # Computing general advantages estimator
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):   
        delta = rewards[step] + gamma * next_value[step] * (1 - done[step]) - values[step]
        gae = delta + (lam * gae)
        returns.insert(0, gae)

    return tf.stack(returns)