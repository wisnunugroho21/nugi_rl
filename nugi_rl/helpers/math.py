import numpy as np
from torch import Tensor

def normalize(data: Tensor, mean = None, std = None, clip = None) -> Tensor:
    if mean is not None and std is not None:
        data_normalized = (data - mean) / (std + 1e-8)
    else:
        data_normalized = (data - data.mean()) / (data.std() + 1e-8)
                
    if clip:
        data_normalized = data_normalized.clamp(-1 * clip, clip)

    return data_normalized 

def prepro_half(I: Tensor) -> Tensor:
    I = I[35:195] # crop
    I = I[::2,::2, 0]
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I

def prepro_crop(I: Tensor) -> Tensor:
    I = I[35:195] 
    return I

def prepo_full(I: Tensor) -> Tensor:
    I = I[35:195] # crop
    I = I[:,:, 0]
    I[I == 144] = 1 # erase background (background type 1)
    I[I == 109] = 1 # erase background (background type 2)
    I[I != 0] = 0 # everything else (paddles, ball) just set to 1
    return I

def prepo_full_one_dim(I: Tensor) -> Tensor:
    I = prepo_full(I)
    I = I.astype(np.float32).ravel()
    I = I / 255.0
    return I

def prepro_half_one_dim(I: Tensor) -> Tensor:
    I = prepro_half(I)
    I = I.float().ravel()
    return I

def prepo_crop(I: Tensor) -> Tensor:
    I = I[35:195] # crop
    I = I / 255.0
    return I

def new_std_from_rewards(rewards, reward_target):
    rewards     = np.array(rewards)
    mean_reward = np.mean(reward_target - rewards)
    new_std     = mean_reward / reward_target

    if new_std < 0.25:
        new_std = 0.25
    elif new_std > 1.0:
        new_std = 1.0

    return new_std

def count_new_mean(prevMean: Tensor, prevLen: Tensor, newData: Tensor) -> Tensor:
    return ((prevMean * prevLen) + newData.sum(0)) / (prevLen + newData.shape[0])
      
def count_new_std(prevStd: Tensor, prevLen: Tensor, newData: Tensor) -> Tensor:
    return (((prevStd.pow(2) * prevLen) + (newData.var(0) * newData.shape[0])) / (prevLen + newData.shape[0])).sqrt()