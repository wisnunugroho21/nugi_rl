from torch import Tensor


class Runner:
    def run(
        self,
    ) -> tuple[
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        raise NotImplementedError
