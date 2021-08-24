import torch
from helpers.pytorch_utils import set_device

class CrossEntropyIndexes():
    def __init__(self, device = torch.device('cuda:0')):
        self.device = device

    def compute_loss(self, logits):
        indexes = torch.arange(logits.shape[0]).long().to(self.device)
        return torch.nn.functional.cross_entropy(logits, indexes)
        