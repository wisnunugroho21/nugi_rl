import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

from nugi_rl.utilities.augmentation.base import Augmentation

class KorniaAugmentation(Augmentation):
    def __init__(self, transform: nn.Sequential) -> None:
        self.transform = transform
        
    @torch.no_grad
    def augment(self, inputs: Tensor) -> Tensor:
        if inputs.max() > 1.0:
            inputs = inputs / 255.0
        
        return self.transform(inputs)