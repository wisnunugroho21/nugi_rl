import torch

class CLR():
    def compute_loss(self, rep_datas_1, rep_datas_2):
        similarity      = torch.nn.functional.cosine_similarity(rep_datas_1, rep_datas_2)
        probs_similar   = torch.nn.functional.log_softmax(similarity)
        return (probs_similar * -1).mean()