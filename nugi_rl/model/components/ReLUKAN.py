import torch
import torch.nn as nn


class ReLUKAN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        g: int = 5,
        k: int = 3,
        order: int = 2,
    ):
        super().__init__()

        self.input_size, self.output_size = input_size, output_size
        self.g, self.k, self.order = g, k, order

        self.r = (2.0 * g / (k + 1.0)) ** 2

        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1.0) / g

        self.phase_low = nn.Parameter(phase_low.repeat(input_size, 1))
        self.phase_high = nn.Parameter(phase_high.repeat(input_size, 1))

        self.out_conv = nn.Conv2d(1, output_size, (g + k, input_size), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)

        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = (x1 * x2 * self.r) ** (self.order)

        x = x.reshape(x.shape[0], 1, self.g + self.k, self.input_size)
        x = self.out_conv(x)
        x = x.reshape(x.shape[0], self.output_size)

        return x
