from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..typing import StandardDeviation


class Sigma(nn.Linear):
    """
    Outputs positive values for standard deviations

    log(sigma^2) = W x + b
    sigma = exp(0.5 * log(sigma^2))
    """

    def __init__(
        self,
        in_features: int,
        n_variables: int,
        bias: bool = True,
        sigma_ref: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, n_variables, bias, device=device, dtype=dtype)
        self.sigma_ref = sigma_ref

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(2 * log(2 * self.sigma_ref))

    def forward(self, input: Tensor) -> StandardDeviation:
        return torch.exp(0.5 * F.linear(input, self.weight, self.bias))
