import torch
import torch.nn as nn
from torch import Tensor

from nugi_rl.model.components.Transformer import DecoderLayer, EncoderLayer


class GlobalExtractor(nn.Module):
    def __init__(self, dim: int, num_layers=2) -> None:
        super().__init__()

        self.object_queries = torch.ones(1, 1, dim)
        self.register_buffer("object_queries", self.object_queries)

        self.encoder = nn.ModuleList(
            [EncoderLayer(dim, heads=4, b=8) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(dim, heads=4, b=8) for _ in range(num_layers)]
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        src_mask = self.transform_mask(mask)
        src_embeddings = inputs
        tgt_embeddings = self.object_queries

        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)

        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings)

        return tgt_embeddings

    def transform_mask(self, mask: Tensor) -> Tensor:
        mask = mask != -1
        return mask.unsqueeze(1).unsqueeze(2)


class Policy_Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
        )

        self.feedforward_3 = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
        )

        self.extractors = GlobalExtractor(64)

        self.feedforward_2 = nn.Sequential(
            nn.Linear(128, 16), nn.GELU(), nn.Linear(16, action_dim), nn.Softmax(-1)
        )

    def forward(self, states: Tensor, detach: bool = False) -> tuple:
        masks = states[0][:, :, -1]

        x = self.feedforward_1(states[0])
        x = self.extractors(x, masks).squeeze(1)

        x2 = self.feedforward_3(states[1])
        x = torch.cat([x, x2], dim=1)

        action = self.feedforward_2(x)

        if detach:
            return (action.detach(),)
        else:
            return action


class Value_Model(nn.Module):
    def __init__(self, state_dim: int):
        super(Value_Model, self).__init__()

        self.feedforward_1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
        )

        self.feedforward_3 = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
        )

        self.extractors = GlobalExtractor(64)

        self.feedforward_2 = nn.Sequential(
            nn.Linear(128, 16), nn.GELU(), nn.Linear(16, 1)
        )

    def forward(self, states: Tensor, detach: bool = False) -> Tensor:
        masks = states[0][:, :, -1]

        x = self.feedforward_1(states[0])
        x = self.extractors(x, masks).squeeze(1)

        x2 = self.feedforward_3(states[1])
        x = torch.cat([x, x2], dim=1)

        value = self.feedforward_2(x)

        if detach:
            return value.detach()
        else:
            return value
