from torch import Tensor

class A2C(): 
    def compute_loss(self, action_datas: tuple, values: Tensor, next_values: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        raise NotImplementedError