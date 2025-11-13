import torch.nn as nn
from torch import Tensor


class Augmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
