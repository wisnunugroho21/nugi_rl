from torch import Tensor


class Augmentation():
    def augment(self, inputs) -> Tensor:
        raise NotImplementedError