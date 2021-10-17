from torch import Tensor

class ValueLoss(): 
    def compute_loss(self, values: Tensor, next_values: Tensor, rewards: Tensor, dones: Tensor, old_values: Tensor = None) -> Tensor:
        raise NotImplementedError