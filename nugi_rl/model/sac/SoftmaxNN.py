import torch
import torch.nn as nn

from torch import Tensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states: Tensor) -> tuple:
      action_datas = self.nn_layer(states)
      return (action_datas, )
      
class Q_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim + 1, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, actions: Tensor) -> tuple:
      if len(actions.shape) == 1:
        actions = actions.unsqueeze(-1)

      x   = torch.cat((states, actions), -1)
      return self.nn_layer(x)