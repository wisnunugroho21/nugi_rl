import tensorflow as tf

def monte_carlo_discounted(reward, done, gamma = 0.99, lam = 0.95):
    returns = []        
    running_add = 0
    
    for i in reversed(range(len(reward))):
        running_add = reward[i] + (1.0 - done) * gamma * running_add  
        returns.insert(0, running_add)
        
    return tf.stack(returns)
    
def temporal_difference(reward, next_value, done, gamma = 0.99, lam = 0.95):
    q_values = reward + (1.0 - done) * gamma * next_value           
    return q_values