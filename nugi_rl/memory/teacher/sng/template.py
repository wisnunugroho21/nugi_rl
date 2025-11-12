from torch import Tensor

from nugi_rl.memory.base import Memory


class SNGTemplateMemory(Memory):
    def __init__(self, capacity=1000000):
        self.capacity = capacity

        self.states: list[Tensor] = []
        self.goals: list[Tensor] = []
        self.next_states: list[Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        return self.states[idx], self.goals[idx], self.next_states[idx]

    def save(
        self,
        state: Tensor,
        goal: Tensor,
        next_state: Tensor,
    ) -> None:
        if len(self) >= self.capacity:
            self.states = self.states[1:]
            self.goals = self.goals[1:]
            self.next_states = self.next_states[1:]

        self.goals.append(goal)

        self.states.append(state)
        self.next_states.append(next_state)

    def save_all(
        self,
        states: list[Tensor],
        goals: list[Tensor],
        next_states: list[Tensor],
    ) -> None:
        for state, goal, next_state in zip(states, goals, next_states):
            self.save(state, goal, next_state)

    def get(
        self, start_position: int = 0, end_position: int | None = None
    ) -> tuple[
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        if end_position is not None and end_position != -1:
            states = self.states[start_position : end_position + 1]
            goals = self.goals[start_position : end_position + 1]
            next_states = self.next_states[start_position : end_position + 1]

        elif start_position is None or start_position < 0:
            states = self.states[:end_position]
            goals = self.goals[:end_position]
            next_states = self.next_states[:end_position]

        else:
            states = self.states[start_position:]
            goals = self.goals[start_position:]
            next_states = self.next_states[start_position:]

        return states, goals, next_states

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
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

            self.goals = [
                *self.goals[:start_position],
                *self.goals[end_position + 1 :],
            ]

            self.next_states = [
                *self.next_states[:start_position],
                *self.next_states[end_position + 1 :],
            ]

        elif end_position is not None and end_position != -1:
            self.states = self.states[end_position + 1 :]
            self.goals = self.goals[end_position + 1 :]
            self.next_states = self.next_states[end_position + 1 :]

        else:
            del self.states
            del self.goals
            del self.next_states

            self.states = []
            self.goals = []
            self.next_states = []
