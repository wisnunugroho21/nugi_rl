import torch.nn as nn
from torch import Tensor

class Ppo(nn.Module): 
    def forward(self, action_datas: tuple, old_action_datas: tuple, actions: Tensor, advantages: Tensor) -> Tensor:
        raise NotImplementedError