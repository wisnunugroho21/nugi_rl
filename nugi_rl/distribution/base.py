from torch import Tensor

class Distribution():
    def sample(self, datas: tuple) -> Tensor:
        raise NotImplementedError
        
    def entropy(self, datas: tuple) -> Tensor:
        raise NotImplementedError
        
    def logprob(self, datas: tuple, value_data: Tensor) -> Tensor:
        raise NotImplementedError

    def kldivergence(self, datas1: tuple, datas2: tuple) -> Tensor:
        raise NotImplementedError

    def deterministic(self, datas: tuple) -> Tensor:
        raise NotImplementedError