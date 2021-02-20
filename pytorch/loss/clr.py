import torch

class CLR():
    def compute_loss(self, first_encoded, second_encoded):
        similarity      = torch.nn.functional.cosine_similarity(first_encoded, second_encoded)
        probs_similar   = torch.nn.functional.log_softmax(similarity, dim = -1)
        return (probs_similar * -1).mean()