import torch
import torch.nn as nn

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(-1)
        )
        
    def forward(self, states, detach = False):
        if detach:
            return self.nn_layer(states).detach()
        else:
            return self.nn_layer(states)

class Value_Model(nn.Module):
    def __init__(self, state_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, states, detach = False):
        if detach:
            return self.nn_layer(states).detach()
        else:
            return self.nn_layer(states)

class RND_Model(nn.Module):
    def __init__(self, state_dim):
        super(RND_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
    def forward(self, states, detach = False):
        if detach:
            return self.nn_layer(states).detach()
        else:
            return self.nn_layer(states)