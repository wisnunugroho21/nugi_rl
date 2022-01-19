import torch
from torch import Tensor

from nugi_rl.loss.clr.base import CLR

class CosineSimilarityCLR(CLR):
    def __init__(self, device = torch.device('cuda')):
        super().__init__()
        
        self.device = device

    def forward(self, first_encoded: Tensor, second_encoded: Tensor) -> Tensor:
        indexes     = torch.arange(first_encoded.shape[0]).long().to(self.device)   
        
        similarity  = torch.nn.functional.cosine_similarity(first_encoded.unsqueeze(1), second_encoded.unsqueeze(0), dim = 2)

        loss1       = torch.nn.functional.cross_entropy(similarity, indexes)
        loss2       = torch.nn.functional.cross_entropy(similarity.t(), indexes)

        return (loss1 + loss2) / 2.0