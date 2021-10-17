from torch import Tensor

class ValueLoss(): 
    def compute_loss(self, values: Tensor, old_values: Tensor, next_values: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        raise NotImplementedError