from torch import Tensor

from nugi_rl.memory.base import Memory


class StateMemory(Memory):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx) -> Tensor:
        raise NotImplementedError

    def save(self, datas: Tensor) -> None:
        raise NotImplementedError

    def save_all(self, datas: list[Tensor]) -> None:
        for data in datas:
            self.save(data)

    def get(
        self, start_position: int = 0, end_position: int | None = None
    ) -> list[Tensor]:
        raise NotImplementedError

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
        raise NotImplementedError
