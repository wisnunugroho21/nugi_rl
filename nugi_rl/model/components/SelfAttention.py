import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, value: Tensor, key: Tensor, query: Tensor) -> Tensor:
        attn_scores         = torch.matmul(query, key.transpose(1, 2))
        scaled_attn_scores  = attn_scores / math.sqrt(query.size(-1))
        attn_scores_softmax = F.softmax(scaled_attn_scores, dim = 1)
        outputs             = torch.matmul(attn_scores_softmax, value)
        
        return outputs