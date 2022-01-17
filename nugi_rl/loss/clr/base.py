from torch import Tensor

class CLR():
    def compute_loss(self, first_encoded: Tensor, second_encoded: Tensor) -> Tensor:
        raise NotImplementedError