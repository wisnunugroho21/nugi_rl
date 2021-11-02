from torch import Tensor

class Distribution():
    def sample(self, datas) -> Tensor:
        raise NotImplementedError
        
    def entropy(self, datas) -> Tensor:
        raise NotImplementedError
        
    def logprob(self, datas, value_data: Tensor) -> Tensor:
        raise NotImplementedError

    def kldivergence(self, datas1, datas2) -> Tensor:
        raise NotImplementedError

    def deterministic(self, datas) -> Tensor:
        raise NotImplementedError