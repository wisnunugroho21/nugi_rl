import torch.nn as nn
from torch import Tensor

class AlphaLoss(nn.Module):
    def forward(self, action_datas: tuple, old_action_datas: tuple, alpha: tuple) -> Tensor:
        raise NotImplementedError