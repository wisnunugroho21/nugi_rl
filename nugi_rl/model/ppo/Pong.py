import torch.nn as nn
from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.nn_layer1 = nn.Sequential(
          nn.Linear(state_dim, 640),
          nn.ReLU()
        )

        self.memory_layer = nn.LSTM(640, 640)

        self.nn_layer2 = nn.Sequential(
          nn.Linear(640, 640),
          nn.ReLU(),
          nn.Linear(640, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states: Tensor) -> tuple:
      x1 = self.nn_layer1(states).transpose(0, 1)

      self.memory_layer.flatten_parameters()

      out_i, _ = self.memory_layer(x1)
      x2 = out_i[-1]

      action_datas = self.nn_layer2(x2)
      return (action_datas, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.nn_layer1 = nn.Sequential(
          nn.Linear(state_dim, 640),
          nn.ReLU()
        )

        self.memory_layer = nn.LSTM(640, 640)

        self.nn_layer2 = nn.Sequential(
          nn.Linear(640, 640),
          nn.ReLU(),
          nn.Linear(640, 640),
          nn.ReLU(),
          nn.Linear(640, 1)
        )
        
    def forward(self, states: Tensor) -> Tensor:
      x1 = self.nn_layer1(states).transpose(0, 1)

      self.memory_layer.flatten_parameters()

      out_i, _ = self.memory_layer(x1)
      x2 = out_i[-1]

      return self.nn_layer2(x2)