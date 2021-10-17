from torch import Tensor

class EntropyLoss():
    def compute_loss(self, action_datas: tuple) -> Tensor:
        raise NotImplementedError