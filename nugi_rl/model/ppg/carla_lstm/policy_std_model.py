import torch
import torch.nn as nn

from helpers.pytorch_utils import set_device

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(PolicyModel, self).__init__()

      self.state_extractor  = nn.Sequential( nn.Linear(2, 32), nn.ReLU() )
      self.image_extractor  = nn.LSTM(256, 128)
      
      self.nn_layer         = nn.Sequential( nn.Linear(160, 192), nn.ReLU() )
      
      self.actor            = nn.Sequential( nn.Linear(64, 2), nn.Tanh() )
      self.std              = nn.Sequential( nn.Linear(64, 2), nn.Sigmoid() )

      self.critic_layer     = nn.Sequential( nn.Linear(64, 1) )
        
    def forward(self, res, state, detach = False):
      out_i, _  = self.image_extractor(res)
      i         = out_i[-1]

      s   = self.state_extractor(state)
      x   = torch.cat([i, s], -1)
      x   = self.nn_layer(x)

      action              = self.actor(x[:, :64])
      std                 = self.std(x[:, 64:128])
      critic              = self.critic_layer(x[:, 128:192])

      if detach:
        return (action.detach(), std.detach()), critic.detach()
      else:
        return (action, std), critic