import torch.nn as nn
from torch import Tensor

class CLR(nn.Module):
    def forward(self, first_encoded: Tensor, second_encoded: Tensor) -> Tensor:
        raise NotImplementedError