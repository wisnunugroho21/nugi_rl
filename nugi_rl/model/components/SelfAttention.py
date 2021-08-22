import torch
import torch.nn as nn
from torch.nn.functional import softmax

class SelfAttention(nn.Module):
    def __init__(self, num_dim):
        super(SelfAttention, self).__init__()

        self.scaling_factor = torch.tensor(num_dim).sqrt()

    def forward(self, value, key, query):
        attn_scores         = query @ key.transpose(1, 2)
        scaled_attn_scores  = attn_scores / self.scaling_factor
        attn_scores_softmax = softmax(scaled_attn_scores, dim = -1)
        outputs             = attn_scores_softmax @ value
        
        return outputs