from torch import Tensor

class AlphaLoss():
    def compute_loss(self, action_datas: tuple, old_action_datas: tuple, alpha: tuple) -> Tensor:
        raise NotImplementedError