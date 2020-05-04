import tensorflow as tf
import numpy as np

def normalize(data):
    data_normalized = (data - tf.math.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-6)
    return data_normalized   

def prepro_half(I):
    I = I[35:195] # crop
    I = I[::2,::2, 0]
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.astype(np.float32).ravel()
    return I