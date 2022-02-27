import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-1 * torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding           = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2]  = torch.sin(pos * den)
        pos_embedding[:, 1::2]  = torch.cos(pos * den)
        pos_embedding           = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.pos_embedding[:tokens.size(0), :]

class MultiHeadAttention(nn.Module):    
    def __init__(self, heads: int, d_model: int):        
        super().__init__()

        assert d_model % heads == 0
        self.d_k        = d_model // heads
        self.heads      = heads
        
        self.query      = nn.Linear(d_model, d_model)
        self.key        = nn.Linear(d_model, d_model)
        self.value      = nn.Linear(d_model, d_model)
        self.out        = nn.Linear(d_model, d_model)
        # self.dropout    = nn.Dropout(0.1)

        self.scl_params = math.sqrt(self.d_k)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        query   = self.query(query)
        key     = self.key(key)        
        value   = self.value(value)
        
        query   = query.view(query.shape[0], self.heads, -1, self.d_k)
        key     = key.view(key.shape[0], self.heads, -1, self.d_k)
        value   = value.view(value.shape[0], self.heads, -1, self.d_k)
       
        scores  = query @ key.transpose(2, 3)
        scores  = scores / self.scl_params
        
        if mask is not None:
            min_type_value  = torch.finfo(scores.dtype).min
            scores  = scores.masked_fill(mask == 0, min_type_value)
             
        weights     = F.softmax(scores, dim = -1)
        # weights     = self.dropout(weights)

        context     = weights @ value
        context     = context.transpose(1, 2).flatten(2)

        interacted  = self.out(context)
        return interacted

class FeedForward(nn.Module):
    def __init__(self, d_model: int, b: int = 4):
        super().__init__()
        
        self.fc1        = nn.Linear(d_model, d_model * b)
        self.fc2        = nn.Linear(d_model * b, d_model)
        # self.dropout    = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, b: int = 4):
        super().__init__()

        self.layernorm1     = nn.LayerNorm(d_model)
        self.layernorm2     = nn.LayerNorm(d_model)

        self.self_multihead = MultiHeadAttention(heads, d_model)
        
        self.feed_forward   = FeedForward(d_model, b = b)
        self.dropout        = nn.Dropout(0.1)

    def forward(self, embeddings: Tensor, mask: Tensor = None) -> Tensor:
        interacted          = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted          = self.layernorm1(interacted + embeddings)

        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        encoded             = self.layernorm2(feed_forward_out + interacted)

        return encoded

class DecoderLayer(nn.Module):    
    def __init__(self, d_model: int, heads: int, b: int = 4):
        super().__init__()

        self.layernorm1     = nn.LayerNorm(d_model)
        self.layernorm2     = nn.LayerNorm(d_model)
        self.layernorm3     = nn.LayerNorm(d_model)

        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead  = MultiHeadAttention(heads, d_model)

        self.feed_forward   = FeedForward(d_model, b = b)
        self.dropout        = nn.Dropout(0.1)
        
    def forward(self, embeddings: Tensor, encoded: Tensor, target_mask: Tensor = None) -> Tensor:
        query               = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query               = self.layernorm1(query + embeddings)

        interacted          = self.dropout(self.src_multihead(query, encoded, encoded, None))
        interacted          = self.layernorm2(interacted + query)

        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        decoded             = self.layernorm3(feed_forward_out + interacted)

        return decoded