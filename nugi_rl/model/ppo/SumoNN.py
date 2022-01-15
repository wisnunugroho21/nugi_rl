import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.Transformer import EncoderLayer, DecoderLayer

class GlobalExtractor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.object_queries = nn.parameter.Parameter(
            torch.ones(1, 1, dim)
        )

        self.encoder_1 = EncoderLayer(dim, heads = 4)
        self.encoder_2 = EncoderLayer(dim, heads = 4)
        self.decoder = DecoderLayer(dim, heads = 4)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        mask = self.transform_mask(mask)

        x = self.encoder_1(inputs, mask)
        x = self.encoder_2(x, mask)
        return self.decoder(self.object_queries, x).squeeze(1)

    def transform_mask(self, mask: Tensor) -> Tensor:
        mask = mask != -100
        return mask.unsqueeze(1).unsqueeze(2)

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, bins: int):
        super(Policy_Model, self).__init__()

        self.action_dim = action_dim
        self.bins = bins

        self.object_queries = nn.parameter.Parameter(
            torch.ones(1, 1, 64)
        )

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
        )

        self.extractors = nn.ModuleList(
            [GlobalExtractor(32) for _ in range(4)]
        )

        self.feedforward_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim * bins),
            nn.Sigmoid()
        )

    def forward(self, states: Tensor, detach: bool = False) -> tuple:
        masks = states.permute(1, 0, 2, 3)[:, :, :, -1]

        datas = self.feedforward_1(states)
        datas = datas.permute(1, 0, 2, 3)

        extracted = []
        for idx, extractor in enumerate(self.extractors):
            extracted.append(extractor(datas[idx], masks[idx]))

        datas = torch.stack(extracted).permute(1, 0, 2).flatten(1)
        
        action = self.feedforward_2(datas)
        action = action.reshape(-1, self.action_dim, self.bins)

        if detach:
            return (action.detach(), )
        else:
            return (action, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
        )

        self.object_queries = nn.parameter.Parameter(
            torch.ones(1, 1, 64)
        )

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
        )

        self.extractors = nn.ModuleList(
            [GlobalExtractor(32) for _ in range(4)]
        )

        self.feedforward_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
        masks = states.permute(1, 0, 2, 3)[:, :, :, -1]

        datas = self.feedforward_1(states)
        datas = datas.permute(1, 0, 2, 3)

        extracted = []
        for idx, extractor in enumerate(self.extractors):
            extracted.append(extractor(datas[idx], masks[idx]))

        datas = torch.stack(extracted).permute(1, 0, 2).flatten(1)

        if detach:
            return self.feedforward_2(datas).detach()
        else:
            return self.feedforward_2(datas)
