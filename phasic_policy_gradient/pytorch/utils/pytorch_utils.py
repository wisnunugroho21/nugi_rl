import torch
import numpy as np

def set_device(use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def to_numpy(datas, use_gpu = True):
    if use_gpu:
        if torch.cuda.is_available():
            return datas.detach().cpu().numpy()
        else:
            return datas.detach().numpy()
    else:
        return datas.detach().numpy()