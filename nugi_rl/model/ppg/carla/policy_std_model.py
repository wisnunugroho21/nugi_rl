import torch
import torch.nn as nn

from helpers.pytorch_utils import set_device

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(PolicyModel, self).__init__()

      self.state_extractor  = nn.Sequential( nn.Linear(2, 32), nn.ReLU() )
      self.image_extractor  = nn.Sequential( nn.Linear(128, 128), nn.ReLU() )
      self.nn_layer         = nn.Sequential( nn.Linear(160, 320), nn.ReLU() )
      
      self.actor_steer      = nn.Sequential( nn.Linear(64, 1) )
      self.actor_gas_break  = nn.Sequential( nn.Linear(64, 1) )

      self.std_steer        = nn.Sequential( nn.Linear(64, 1), nn.Sigmoid() )
      self.std_gas_break    = nn.Sequential( nn.Linear(64, 1), nn.Sigmoid() )

      self.critic_layer     = nn.Sequential( nn.Linear(64, 1) )
        
    def forward(self, res, state, detach = False):
      i   = self.image_extractor(res)
      s   = self.state_extractor(state)
      x   = torch.cat([i, s], -1)
      x   = self.nn_layer(x)

      action_steer        = self.actor_steer(x[:, :64])
      std_steer           = self.std_steer(x[:, 64:128])

      action_gas_break    = self.actor_gas_break(x[:, 128:192])
      std_gas_break       = self.std_gas_break(x[:, 192:256])

      action              = torch.cat((action_steer, action_gas_break), -1)
      std                 = torch.cat((std_steer, std_gas_break), -1)
      critic              = self.critic_layer(x[:, 256:320])

      if detach:
        return (action.detach(), std.detach()), critic.detach()
      else:
        return (action, std), critic