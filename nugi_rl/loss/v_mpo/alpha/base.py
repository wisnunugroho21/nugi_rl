import torch.nn as nn
from torch import Tensor


class AlphaLoss(nn.Module):
    def forward(
        self, action_datas: Tensor, old_action_datas: Tensor, alpha: Tensor
    ) -> Tensor:
        raise NotImplementedError
