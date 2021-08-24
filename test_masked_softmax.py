import torch
from torch.nn.functional import softmax

def masked_softmax(input: torch.Tensor, bool_mask: torch.Tensor, dim: int = -1, dtype: torch.dtype = None) -> torch.Tensor:
    min_type_value = torch.finfo(input.dtype).min
    masked_value = input.masked_fill(bool_mask, min_type_value)

    return softmax(masked_value, dim = dim, dtype = dtype)
    
a = torch.rand(1, 3)
print(a)

b = masked_softmax(a, a < 0.5)
print(b)

c = masked_softmax(a, torch.tensor([True, False, False]))
print(c)