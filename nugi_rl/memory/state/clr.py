from copy import deepcopy

import torch
import torchvision.transforms as transforms
from memory.state.base import StateMemory
from torch import Tensor
from torchvision.transforms import Compose


class ClrMemory(StateMemory):
    def __init__(
        self,
        input_trans: Compose | None = None,
        target_trans: Compose | None = None,
        capacity: int = 10000,
    ):
        self.images: list[Tensor] = []
        self.capacity = capacity

        if input_trans is None:
            self.input_trans = transforms.Compose(
                [
                    transforms.RandomResizedCrop(320),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(33),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.input_trans = input_trans

        if target_trans is None:
            self.target_trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.target_trans = target_trans

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tensor:
        images = self.images[idx]

        input_images = self.input_trans(images)
        target_images = self.target_trans(images)

        return torch.stack([input_images, target_images])

    def save(self, datas: Tensor) -> None:
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(datas)

    def get(
        self, start_position: int = 0, end_position: int | None = None
    ) -> list[Tensor]:
        if end_position is not None or end_position == -1:
            images = self.images[start_position : end_position + 1]
        else:
            images = self.images[start_position:]

        return images

    def clear(self, start_position: int = 0, end_position: int | None = None) -> None:
        if end_position is not None and end_position != -1:
            del self.images[start_position : end_position + 1]
        else:
            del self.images[start_position:]
