from torch import Tensor

from nugi_rl.memory.state.base import StateMemory


class PPGMemory(StateMemory):
    def __init__(self, capacity=1000000):
        self.states: list[Tensor] = []
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx) -> Tensor:
        return self.states[idx]

    def save(self, datas: Tensor) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]

        self.states.append(datas)

    def get(
        self, start_position: int | None = 0, end_position: int | None = None
    ) -> list[Tensor]:
        if end_position is not None and end_position != -1:
            if start_position is None or start_position < 0:
                states = self.states[: end_position + 1]
            else:
                states = self.states[start_position : end_position + 1]

        else:
            states = self.states[start_position:]

        return states

    def clear(
        self, start_position: int | None = 0, end_position: int | None = None
    ) -> None:
        if (
            start_position is not None
            and start_position > 0
            and end_position is not None
            and end_position != -1
        ):
            self.states = [
                *self.states[:start_position],
                *self.states[end_position + 1 :],
            ]

        elif start_position is not None and start_position > 0:
            self.states = self.states[:start_position]

        elif end_position is not None and end_position != -1:
            self.states = self.states[end_position + 1 :]

        else:
            del self.states
            self.states = []
