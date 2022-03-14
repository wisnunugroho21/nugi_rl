from torch import Tensor
from PIL.Image import Image
import torch.nn as nn

from typing import Union

class Augmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs: Union[Image, Tensor]) -> Tensor:
        raise NotImplementedError