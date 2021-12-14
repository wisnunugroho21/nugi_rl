import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.SelfAttention import SelfAttention

class Encoder(nn.Module):
    def __init__(self, dim) -> None:
        super(Encoder, self).__init__()

        self.value  = nn.Linear(dim, dim)
        self.key    = nn.Linear(dim, dim)
        self.query  = nn.Linear(dim, dim)

        self.att    = SelfAttention()

        self.feedforward = nn.Sequential(
          nn.Linear(dim, dim),
          nn.SiLU()
        )

    def forward(self, datas: Tensor, mask: Tensor) -> Tensor:
        value     = self.value(datas)
        key       = self.key(datas)
        query     = self.query(datas)

        context   = self.att(value, key, query, mask)
        context   = context + datas

        encoded   = self.feedforward(context)
        return encoded + context

class FinalEncoder(nn.Module):
    def __init__(self, dim) -> None:
        super(FinalEncoder, self).__init__()

        self.value  = nn.Linear(dim, dim)
        self.key    = nn.Linear(dim, dim)
        self.query  = nn.parameter.Parameter(
          torch.ones(1, 1, dim)
        )

        self.att    = SelfAttention()

        self.feedforward = nn.Sequential(
          nn.Linear(dim, dim),
          nn.SiLU()
        )

    def forward(self, datas: Tensor, mask: Tensor) -> Tensor:
        value     = self.value(datas)
        key       = self.key(datas)

        context   = self.att(value, key, self.query, mask).squeeze(1)

        encoded   = self.feedforward(context)
        return encoded + context

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bins: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins       = bins

        self.feedforward_1 = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.SiLU(),
        )

        self.encoder_1 = Encoder(32)
        self.encoder_2 = Encoder(32)
        self.encoder_3 = FinalEncoder(32)

        self.feedforward_2 = nn.Sequential(
          nn.Linear(32, 64),
          nn.SiLU(),
          nn.Linear(64, action_dim * bins),
          nn.Sigmoid()
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> tuple:
      datas = self.feedforward_1(states)

      datas = self.encoder_1(datas, states[:, :, -1].unsqueeze(1))
      datas = self.encoder_2(datas, states[:, :, -1].unsqueeze(1))
      datas = self.encoder_3(datas, states[:, :, -1].unsqueeze(1))

      action = self.feedforward_2(datas)
      action = action.reshape(-1, self.action_dim, self.bins)

      if detach:
        return (action.detach(), )
      else:
        return (action, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
          nn.Linear(state_dim, 32),
          nn.SiLU(),
        )

        self.encoder_1 = Encoder(32)
        self.encoder_2 = Encoder(32)
        self.encoder_3 = FinalEncoder(32)

        self.feedforward_2 = nn.Sequential(
          nn.Linear(32, 64),
          nn.SiLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
      datas = self.feedforward_1(states)

      datas = self.encoder_1(datas, states[:, :, -1].unsqueeze(1))
      datas = self.encoder_2(datas, states[:, :, -1].unsqueeze(1))
      datas = self.encoder_3(datas, states[:, :, -1].unsqueeze(1))

      if detach:
        return self.feedforward_2(datas).detach()
      else:
        return self.feedforward_2(datas)