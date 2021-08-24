import torch
from helpers.pytorch_utils import set_device

class Moco():
    def __init__(self, device = torch.device('cuda:0')):
        self.device = device

    def compute_loss(self, first_encoded, second_encoded):
        indexes     = torch.arange(first_encoded.shape[0]).long().to(self.device)   
        
        similarity  = torch.mm(first_encoded, second_encoded.t())
        
        loss1       = torch.nn.functional.cross_entropy(similarity, indexes)
        loss2       = torch.nn.functional.cross_entropy(similarity.t(), indexes)

        return (loss1 + loss2) / 2.0
        