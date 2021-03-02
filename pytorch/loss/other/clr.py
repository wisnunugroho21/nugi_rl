import torch
from utils.pytorch_utils import set_device

class CLR():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        zeros           = torch.zeros(first_encoded.shape[0]).long().to(set_device(self.use_gpu))

        first_encoded   = ((first_encoded - first_encoded.mean()) / (first_encoded.std() + 1e-6))
        second_encoded  = ((second_encoded - second_encoded.mean()) / (second_encoded.std() + 1e-6))

        similarity      = torch.nn.functional.cosine_similarity(first_encoded.unsqueeze(1), second_encoded.unsqueeze(0), dim = 2)
        return torch.nn.functional.cross_entropy(similarity, zeros)