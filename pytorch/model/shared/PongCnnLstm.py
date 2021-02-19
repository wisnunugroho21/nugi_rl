import torch
import torch.nn as nn
from utils.pytorch_utils import set_device

class Resnet(nn.Module):
  def __init__(self, use_gpu = True):
    super(Resnet, self).__init__() 

    self.input_layer_1 = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_1_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_2 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_2_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_3 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_3_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_4 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_4_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_post_layer = nn.Sequential(
      nn.ReLU(),
      nn.Flatten(),        
      nn.Linear(800, 200),
      nn.ReLU())

  def forward(self, states):  
      x         = states

      x         = self.input_layer_1(x)
      x1        = self.input_layer_1_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_2(x)
      x1        = self.input_layer_2_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_3(x)
      x1        = self.input_layer_3_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_4(x)
      x1        = self.input_layer_4_1(x)
      x         = torch.add(x, x1)

      x         = self.input_post_layer(x)
      return x

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()   

      self.use_gpu = use_gpu

      self.input_layer = Resnet().float().to(set_device(use_gpu))
      self.memory_layer = nn.LSTM(200, 200).float().to(set_device(use_gpu))

      self.actor_layer = nn.Sequential(
        nn.Linear(200, 25),
        nn.ReLU(),
        nn.Linear(25, action_dim),
        nn.Softmax(-1)
      ).float().to(set_device(use_gpu))

      self.critic_layer = nn.Sequential(
        nn.Linear(200, 25),
        nn.ReLU(),
        nn.Linear(25, 1)
      ).float().to(set_device(use_gpu))        
        
    def forward(self, states, is_actor = False, is_critic = False):  
      batch_size, timesteps, C,H, W = states.size()

      c_in  = states.view(batch_size * timesteps, C, H, W)
      c_out = self.input_layer(c_in)

      r_in              = c_out.view(-1, batch_size, c_out.shape[-1])
      hidden            = torch.ones(1, batch_size, c_out.shape[-1]).float().to(set_device(self.use_gpu))
      r_out, (h_n, h_c) = self.memory_layer(r_in, (hidden, hidden))

      last_in = r_out[-1]

      if is_actor and not is_critic:
        return self.actor_layer(last_in)
      elif is_critic and not is_actor:
        return self.critic_layer(last_in)
      else:
        return self.actor_layer(last_in), self.critic_layer(last_in)