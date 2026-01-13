from typing import Optional

import torch
from torch import Tensor, dtype, nn
from torch.types import Device

from .Sigma import Sigma
from .VariationalLayer import VariationalLayer


class IsotropicGaussian(nn.Module, VariationalLayer):
    def __init__(
        self,
        in_features: int,
        n_variables: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        factory_kwargs = {"bias": bias, "device": device, "dtype": dtype}
        super().__init__()

        self.mu_layer = nn.Linear(in_features, n_variables, **factory_kwargs)
        self.sigma_layer = Sigma(
            in_features, n_variables, sigma_ref=1.0, **factory_kwargs
        )

        self.mu: Optional[Tensor] = None
        self.sigma: Optional[Tensor] = None

    def forward(self, input) -> Tensor:
        mu = self.mu_layer(input)
        sigma = self.sigma_layer(input)

        if self.training:
            self.mu, self.sigma = mu, sigma

        return self._reparameterization(mu, sigma)

    def _reparameterization(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Reparameterization trick: z ~ N(mu, sigma^2 I)

        1. Sample eps ~ N(0, I)
        2. Return: z = mu + sigma * eps

        Note:
            Uses a single sample per data point (L=1) as suggested by Kingma & Welling, relying on a sufficiently
            large batch size. However, this could be extended to multiple samples if needed.

        Args:
            mu (Tensor): Mean of the Gaussian [batch_size, n_variables]
            sigma (Tensor): Standard deviation of the Gaussian [batch_size, n_variables]
        Returns:
            Tensor: Sampled latent variable z [batch_size, n_variables]
        """
        eps = torch.normal(0, torch.ones_like(sigma))
        return mu + sigma * eps

    def kl_reg(self) -> Tensor:
        """
        KL-Divergence regularization between two isotropic Gaussians:
        KL(N(mu_q, sigma_q^2 I) || N(0, I))

        Returns:
            Tensor: KL divergence for each data point [batch_size]
        """
        if self.mu is None or self.sigma is None:
            raise RuntimeError(
                "forward() must be called before kl_reg(). "
                "mu and sigma are not initialized."
            )

        return 0.5 * torch.sum(
            self.sigma**2 + self.mu**2 - 1 - torch.log(self.sigma**2 + 1e-8), dim=1
        )
