import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import math

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = True, depth_multiplier = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.nn_layer(x)

class MultiHeadAttention(nn.Module):    
    def __init__(self, heads: int, d_model: int):        
        super(MultiHeadAttention, self).__init__()

        self.d_k    = d_model // heads
        self.heads  = heads

        self.dropout    = nn.Dropout(0.1)
        self.query      = nn.Linear(d_model, d_model)
        self.key        = nn.Linear(d_model, d_model)
        self.value      = nn.Linear(d_model, d_model)
        self.out        = nn.Linear(d_model, d_model)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        query   = self.query(query)
        key     = self.key(key)        
        value   = self.value(value)   
        
        query   = query.view(query.shape[0], self.heads, -1, self.d_k)
        key     = key.view(key.shape[0], self.heads, -1, self.d_k)
        value   = value.view(value.shape[0], self.heads, -1, self.d_k)
       
        scores      = torch.matmul(query, key.transpose(2, 3))
        scores      = scores / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores  = scores.masked_fill(mask == 0, -1e9)
             
        weights     = F.softmax(scores, dim = -1)
        weights     = self.dropout(weights)

        context     = torch.matmul(weights, value)
        context     = context.transpose(1, 2).flatten(2)

        interacted  = self.out(context)
        return interacted

class LocalPerceptionUnit(nn.Module):
    def __init__(self, dim):
        super(LocalPerceptionUnit, self).__init__()

        self.covnet = DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        x1  = self.covnet(x)
        x   = x + x1

        return x

class InvertedResidualFFN(nn.Module):
    def __init__(self, dim):
        super(InvertedResidualFFN, self).__init__()

        self.covnet1 = nn.Conv2d(dim, dim, kernel_size = 1)
        self.covnet2 = DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, padding = 1)
        self.covnet3 = nn.Conv2d(dim, dim, kernel_size = 1)

        self.bn1    = nn.BatchNorm2d(dim)
        self.bn2    = nn.BatchNorm2d(dim)
        self.bn3    = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        x   = self.covnet1(x)
        x   = self.bn1(F.gelu(x))

        x1  = self.covnet2(x)
        x   = self.bn2(F.gelu(x + x1))

        x   = self.covnet3(x)
        x   = self.bn3(x)

        return x

class LMHSA(nn.Module):
    def __init__(self, dim, k, head):
        super(LMHSA, self).__init__()

        self.mhsa           = MultiHeadAttention(head, dim)
        self.key_covnet     = DepthwiseSeparableConv2d(dim, dim, kernel_size = k, stride = k)
        self.value_covnet   = DepthwiseSeparableConv2d(dim, dim, kernel_size = k, stride = k)

    def forward(self, x: Tensor):
        b, c, h, w = x.shape

        key = self.key_covnet(x)
        val = self.value_covnet(x)
        qry = x

        key = key.permute(0, 2, 3, 1).flatten(1, 2)
        val = val.permute(0, 2, 3, 1).flatten(1, 2)
        qry = qry.permute(0, 2, 3, 1).flatten(1, 2)

        out = self.mhsa(qry, key, val)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return out

class CMTBlock(nn.Module):
    def __init__(self, dim, k, head):
        super(CMTBlock, self).__init__()

        self.lpu    = LocalPerceptionUnit(dim)
        self.lmhsa  = LMHSA(dim, k, head)
        self.irffn  = InvertedResidualFFN(dim)

    def forward(self, x):
        x = self.lpu(x)

        x1 = F.layer_norm(x, [x.shape[-1], x.shape[-2]])
        x1 = self.lmhsa(x1)
        x1 = x + x1

        x2 = F.layer_norm(x1, [x1.shape[-1], x1.shape[-2]])
        x2 = self.irffn(x2)
        x2 = x1 + x2

        return x2

class CMTStem(nn.Module):
    def __init__(self, dim):
        super(CMTStem, self).__init__()

        self.covnet = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = 3, stride = 2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size = 3, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size = 3, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.covnet(x)