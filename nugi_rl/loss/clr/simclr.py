import torch
from helpers.pytorch_utils import set_device

class SimCLR():
    def __init__(self, device = torch.device('cuda:0')):
        self.device = device

    def compute_loss(self, first_encoded, second_encoded):
        indexes     = torch.arange(first_encoded.shape[0]).long().to(self.device)   
        
        similarity  = torch.nn.functional.cosine_similarity(first_encoded.unsqueeze(1), second_encoded.unsqueeze(0), dim = 2)

        loss1       = torch.nn.functional.cross_entropy(similarity, indexes)
        loss2       = torch.nn.functional.cross_entropy(similarity.t(), indexes)

        return (loss1 + loss2) / 2.0