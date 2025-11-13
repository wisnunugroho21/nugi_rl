import torch.nn as nn
from torch import Tensor


class CqlRegularizer(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()

        self.alpha = alpha

    def forward(
        self,
        predicted_q1: Tensor,
        predicted_q2: Tensor,
        naive_q1_value: Tensor,
        naive_q2_value: Tensor,
    ) -> Tensor:
        cql_regularizer1 = ((naive_q1_value - predicted_q1) * self.alpha).mean()
        cql_regularizer2 = ((naive_q2_value - predicted_q2) * self.alpha).mean()

        return cql_regularizer1 + cql_regularizer2
