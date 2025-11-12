import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nugi_rl.loss.adversarial_motion_priors import DiscriminatorLoss
from nugi_rl.memory.teacher.sng.standard import SNGMemory
from nugi_rl.teacher.base import Teacher


class TeacherAdvMtnPrior(Teacher):
    def __init__(
        self,
        discrim: Module,
        discrim_loss: DiscriminatorLoss,
        memory: SNGMemory,
        optimizer: Optimizer,
        epochs: int = 10,
        is_training_mode: bool = True,
        batch_size: int = 32,
        folder: str = "model",
        device: device = torch.device("cuda:0"),
        dont_unsqueeze=False,
    ) -> None:
        self.dont_unsqueeze = dont_unsqueeze
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training_mode = is_training_mode
        self.folder = folder

        self.discrim = discrim
        self.memory = memory
        self.discrim_loss = discrim_loss

        self.optimizer = optimizer
        self.device = device

        if is_training_mode:
            self.discrim.train()
        else:
            self.discrim.eval()

    def _update_step(
        self,
        expert_states: Tensor,
        expert_next_states: Tensor,
        policy_states: Tensor,
        policy_next_states: Tensor,
        goals: Tensor,
    ) -> None:
        self.optimizer.zero_grad()

        dis_expert = self.discrim(expert_states, expert_next_states, goals)
        dis_policy = self.discrim(policy_states, policy_next_states, goals)

        loss = self.discrim_loss(
            dis_expert, dis_policy, policy_states, policy_next_states
        )

        loss.backward()
        self.optimizer.step()

    def teach(self, state: Tensor, next_state: Tensor, goal: Tensor) -> Tensor:
        with torch.inference_mode():
            state = state if self.dont_unsqueeze else state.unsqueeze(0)
            next_state = next_state if self.dont_unsqueeze else next_state.unsqueeze(0)
            goal = goal if self.dont_unsqueeze else goal.unsqueeze(0)

            discrimination = self.discrim(state, next_state, goal)
            reward = torch.max(
                torch.tensor([0]), 1 - 0.25 * (discrimination - 1).pow(2)
            )

            reward = reward.squeeze(0)

        return reward

    def save_obs(self, state: Tensor, goal: Tensor, next_state: Tensor) -> None:
        self.memory.save(state, goal, next_state)

    def save_all(
        self, states: list[Tensor], goals: list[Tensor], next_states: list[Tensor]
    ) -> None:
        self.memory.save_all(states, goals, next_states)

    def update(self) -> None:
        for _ in range(self.epochs):
            dataloader = DataLoader(self.memory, self.batch_size, shuffle=False)
            for (
                expert_states,
                expert_next_states,
                policy_states,
                policy_next_states,
                goals,
            ) in dataloader:
                self._update_step(
                    expert_states,
                    expert_next_states,
                    policy_states,
                    policy_next_states,
                    goals,
                )

        self.memory.clear()

    def get_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> tuple:
        return self.memory.get_policy(start_position, end_position)

    def clear_obs(
        self, start_position: int = 0, end_position: int | None = None
    ) -> None:
        self.memory.clear(start_position, end_position)

    def load_weights(self) -> None:
        model_checkpoint = torch.load(
            self.folder + "/ppo.tar", map_location=self.device
        )
        self.discrim.load_state_dict(model_checkpoint["discrim_model_state_dict"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(model_checkpoint["ppo_optimizer_state_dict"])

        if self.is_training_mode:
            self.discrim.train()
        else:
            self.discrim.eval()

    def save_weights(self) -> None:
        torch.save(
            {
                "discrim_model_state_dict": self.discrim.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.folder + "/amp.tar",
        )
