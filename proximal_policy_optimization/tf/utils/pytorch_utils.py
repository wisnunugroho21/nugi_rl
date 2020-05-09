import torch
import numpy as np

def set_device(use_gpu = True):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def to_numpy(datas, use_gpu = True):
    if use_gpu:
        return datas.cpu().numpy()
    else:
        return datas.numpy()