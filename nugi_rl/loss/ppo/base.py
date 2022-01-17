from torch import Tensor

class Ppo(): 
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, actions: Tensor, advantages: Tensor) -> Tensor:
        raise NotImplementedError