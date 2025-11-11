import torch.nn as nn
from torch import Tensor


class Ppo(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        action_datas: Tensor,
        old_action_datas: Tensor,
        actions: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        raise NotImplementedError
