from torch import Tensor


class Distribution:
    def sample(self, datas: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self, datas: Tensor) -> Tensor:
        raise NotImplementedError

    def logprob(self, datas: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError

    def kldivergence(self, datas1: Tensor, datas2: Tensor) -> Tensor:
        raise NotImplementedError

    def deterministic(self, data: Tensor) -> Tensor:
        raise NotImplementedError
