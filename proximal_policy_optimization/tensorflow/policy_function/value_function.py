import tensorflow as tf

def monte_carlo_discounted(datas, dones, gamma = 0.99, lam = 0.95):
    # Discounting future reward        
    returns = []        
    running_add = 0

    for i in reversed(range(len(datas))):
        running_add = datas[i] + gamma * running_add * (1 - dones)
        returns.insert(0, running_add)

    return tf.stack(returns)

def temporal_difference(rewards, next_values, dones, gamma = 0.99, lam = 0.95):
    # Finding TD Values
    # TD = R + V(St+1)
    TD = rewards + gamma * next_values * (1 - dones)        
    return TD