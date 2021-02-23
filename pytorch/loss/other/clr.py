import torch
from utils.pytorch_utils import set_device

class CLR():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        first_encoded   = torch.nn.functional.normalize(first_encoded, dim = 1)
        second_encoded  = torch.nn.functional.normalize(second_encoded, dim = 1)

        representations = torch.cat((first_encoded, second_encoded), dim = 0)
        similarity      = torch.nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim = 2)

        zeros           = torch.zeros(similarity.shape[0]).long().to(set_device(self.use_gpu))
        return torch.nn.functional.cross_entropy(similarity, zeros)