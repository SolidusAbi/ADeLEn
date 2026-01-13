from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from ..typing import Mean, StandardDeviation
from .VariationalLayer import VariationalLayer


class AnomalyDetector(nn.Linear, VariationalLayer):
    """
    This implementation is for anomaly detection.
    """

    def __init__(self, in_features, out_features, bias=True, sigma_anomaly=3) -> None:
        super(AnomalyDetector, self).__init__(in_features, out_features, bias)
        self.log_sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_bias = Parameter(torch.Tensor(out_features))

        # torch.nn.init.xavier_normal_(self.log_sigma_weight)
        torch.nn.init.xavier_uniform_(self.log_sigma_weight)
        self.log_sigma_bias.data.fill_(2 * np.log((2 * sigma_anomaly)))

        self.sigma_anomaly = sigma_anomaly
        self.mu: Optional[Mean] = None
        self.sigma: Optional[StandardDeviation] = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight, self.bias)
        sigma = torch.exp(
            0.5 * F.linear(input, self.log_sigma_weight, self.log_sigma_bias)
        )

        # if self.training:
        self.mu, self.sigma = mu, sigma

        # Reparameterization trick
        eps = torch.normal(0, torch.ones_like(sigma))
        return mu + sigma * eps

    def kl_reg(self, targets: torch.Tensor) -> torch.Tensor:
        # KL-Divergence regularization
        if self.mu is None or self.sigma is None:
            raise RuntimeError(
                "forward() must be called before kl_reg(). "
                "mu and sigma are not initialized."
            )

        assert torch.all((targets == 0) | (targets == 1))
        sigma_2 = (
            (torch.ones_like(targets) + ((self.sigma_anomaly - 1) * targets))
            .unsqueeze(1)
            .cuda()
        )
        result = (
            torch.log(sigma_2)
            - torch.log(self.sigma)
            + (self.sigma**2 + self.mu**2) / (2 * sigma_2**2)
            - 0.5
        )

        return result.mean()
