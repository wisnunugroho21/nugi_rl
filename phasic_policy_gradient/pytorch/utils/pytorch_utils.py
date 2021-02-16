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

def to_tensor(datas, use_gpu = True, first_unsqueeze = False, last_unsqueeze = False):
    if isinstance(datas, tuple):
        datas = list(datas)
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                datas[i]    = data.unsqueeze(0).detach() if len(data.shape) == 1 or len(data.shape) == 3 else data.detach()
            if last_unsqueeze:
                datas[i]    = data.unsqueeze(-1).detach() if data.shape[-1] != 1 else data.detach()
        datas = tuple(datas)

    elif isinstance(datas, list):
        for i, data in enumerate(datas):
            data    = torch.FloatTensor(data).to(set_device(use_gpu))
            if first_unsqueeze: 
                datas[i]    = data.unsqueeze(0).detach() if len(data.shape) == 1 or len(data.shape) == 3 else data.detach()
            if last_unsqueeze:
                datas[i]    = data.unsqueeze(-1).detach() if data.shape[-1] != 1 else data.detach()
        datas = tuple(datas)

    else:
        datas   = torch.FloatTensor(datas).to(set_device(use_gpu))
        if first_unsqueeze: 
            datas   = datas.unsqueeze(0).detach() if len(datas.shape) == 1 or len(datas.shape) == 3 else datas.detach()
        if last_unsqueeze:
            datas   = datas.unsqueeze(-1).detach() if datas.shape[-1] != 1 else datas.detach()

    return datas
