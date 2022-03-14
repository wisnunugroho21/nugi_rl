from torch import Tensor
from PIL.Image import Image

from typing import Union
from torchvision.transforms.transforms import Compose

from nugi_rl.utilities.augmentation.base import Augmentation

class TorchVisionAugmentation(Augmentation):
    def __init__(self, transform: Compose) -> None:
        super().__init__()
        self.transform = transform
        
    def forward(self, inputs: Union[Image, Tensor]) -> Tensor:
        return self.transform(inputs)