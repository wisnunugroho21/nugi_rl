import torch
import torch.nn as nn


class ReLUKAN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        g: int = 10,
        k: int = 3,
        imin: float = 0.0,
        imax: float = 10.0,
        order: int = 4,
    ):
        super().__init__()

        self.input_size, self.output_size = input_size, output_size
        self.g, self.k, self.order = g, k, order

        self.r = (2.0 * g / (k + 1.0) * (imax - imin)) ** 2

        phase_low = (imax - imin) * torch.arange(-k, g) / g + imin
        phase_high = phase_low + (k + 1.0) / g * (imax - imin)

        self.phase_low = nn.Parameter(phase_low.repeat(input_size, 1).unsqueeze(0))
        self.phase_high = nn.Parameter(phase_high.repeat(input_size, 1).unsqueeze(0))
        self.out_conv = nn.Conv2d(1, output_size, (input_size, (g + k)), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).expand(-1, -1, self.phase_low.shape[2])

        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = (x1 * x2 * self.r) ** (self.order)

        x = x.reshape(x.shape[0], 1, self.input_size, (self.g + self.k))
        x = self.out_conv(x)
        x = x.reshape(x.shape[0], self.output_size)

        return x
