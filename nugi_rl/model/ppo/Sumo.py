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
        return self.decoder(self.object_queries, x)

    def transform_mask(self, mask: Tensor) -> Tensor:
        mask = mask != -1
        return mask.unsqueeze(1).unsqueeze(2)

class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.extractors = nn.ModuleList(
            [GlobalExtractor(64) for _ in range(4)]
        )

        self.feedforward_2 = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(-1)
        )

    def forward(self, states: Tensor, detach: bool = False) -> tuple:
        masks = states.permute(1, 0, 2, 3)[:, :, :, -1]

        datas = self.feedforward_1(states)
        datas = datas.permute(1, 0, 2, 3)

        extracted = [extractor(data, mask) for extractor, data, mask in zip(self.extractors, datas, masks)]

        datas = torch.stack(extracted).squeeze(2).permute(1, 0, 2).flatten(1)        
        action = self.feedforward_2(datas)

        if detach:
            return (action.detach(), )
        else:
            return (action, )

class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        self.extractors = nn.ModuleList(
            [GlobalExtractor(64) for _ in range(4)]
        )

        self.feedforward_2 = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
        masks = states.permute(1, 0, 2, 3)[:, :, :, -1]

        datas = self.feedforward_1(states)
        datas = datas.permute(1, 0, 2, 3)

        extracted = [extractor(data, mask) for extractor, data, mask in zip(self.extractors, datas, masks)]

        datas = torch.stack(extracted).squeeze(2).permute(1, 0, 2).flatten(1)

        if detach:
            return self.feedforward_2(datas).detach()
        else:
            return self.feedforward_2(datas)
