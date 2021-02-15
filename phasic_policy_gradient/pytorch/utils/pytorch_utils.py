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

def to_tensor(datas, use_gpu = True, unsqueeze = 0):
    if isinstance(datas, tuple):
        datas = list(datas)
        for i, s in enumerate(list(datas)):
            s           = torch.FloatTensor(s).to(set_device(use_gpu))
            datas[i]    = s.unsqueeze(0).detach() if len(s.shape) == 1 or len(s.shape) == 3 else s.detach()
        datas = tuple(datas)            
    else:
        state   = torch.FloatTensor(state).to(self.device)
        state   = state.unsqueeze(0).detach() if len(state.shape) == 1 or len(state.shape) == 3 else state.detach()
