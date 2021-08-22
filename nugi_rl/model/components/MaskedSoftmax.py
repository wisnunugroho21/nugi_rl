import torch
import torch.nn as nn
from torch.nn.functional import softmax

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()

    def forward(self, input: torch.Tensor, bool_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        min_type_value  = torch.finfo(input.dtype).min
        masked_value    = input.masked_fill(bool_mask, min_type_value)

        return softmax(masked_value, dim = dim)