from torch import Tensor
from torchvision.transforms.transforms import Compose

from nugi_rl.utilities.augmentation.base import Augmentation

class TorchVisionAugmentation(Augmentation):
    def __init__(self, transform: Compose) -> None:
        self.transform = transform
        
    def augment(self, inputs: Tensor) -> Tensor:
        return self.transform(inputs)