import tensorflow as tf
from utils.pytorch_utils import set_device

def generalized_advantage_estimation(rewards, values, next_values, dones, gamma = 0.99, lam = 0.95):
    gae     = 0
    adv     = []     

    delta   = rewards + (1.0 - dones) * gamma * next_values - values          
    for step in reversed(range(len(rewards))):  
        gae = delta[step] + (1.0 - dones[step]) * gamma * lam * gae
        adv.insert(0, gae)
        
    return tf.stack(adv)