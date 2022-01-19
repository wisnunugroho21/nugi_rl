import torch.nn as nn
from torch import Tensor

class RndStatePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, state_pred: Tensor, state_target: Tensor) -> Tensor:
        state_target = state_target.detach()
        forward_loss = ((state_target - state_pred).pow(2) * 0.5).mean()
        return forward_loss