from typing import Optional

import torch
from torch import Tensor, nn

from ..typing import CholeskyMatrix, Mean, StandardDeviation
from .Cholesky import Cholesky
from .Sigma import Sigma
from .VariationalLayer import VariationalLayer


class Gaussian(nn.Module, VariationalLayer):
    def __init__(
        self,
        in_features: int,
        n_variables: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"bias": bias, "device": device, "dtype": dtype}
        super().__init__()
        self.n_variables = n_variables

        self.mu_layer = nn.Linear(in_features, n_variables, **factory_kwargs)
        self.sigma_layer = Sigma(
            in_features, n_variables, sigma_ref=1.0, **factory_kwargs
        )
        self.cholesky_layer = Cholesky(in_features, n_variables, **factory_kwargs)

        self.mu: Optional[Mean] = None
        self.sigma: Optional[StandardDeviation] = None

    def forward(self, input: Tensor) -> torch.Tensor:
        mu: Mean = self.mu_layer(input)
        sigma: StandardDeviation = self.sigma_layer(input)
        L: CholeskyMatrix = self.cholesky_layer(input)
        L_corr = L / torch.norm(L, p=2, dim=-1, keepdim=True)

        # Save for kl_reg
        self.mu = mu
        self.sigma = sigma
        self.L_corr = L_corr

        return self._reparameterization(mu, sigma, L_corr)

    def _reparameterization(
        self, mu: Mean, sigma: StandardDeviation, L_corr: CholeskyMatrix
    ) -> Tensor:
        """
        Reparameterization trick: z ~ N(mu, Sigma)

        1. Sample eps ~ N(0, I)
        2. Compute correlated noise: z_corr = L_corr @ eps
        3. Return: z = mu + sigma * z_corr

        Note:
            Uses a single sample per data point (L=1) as suggested by Kingma & Welling, relying on a sufficiently
            large batch size. However, this could be extended to multiple samples if needed.

        Args:
            mu (Mean): Mean vector of shape (B, N)
            sigma (StandardDeviation): Standard deviation vector of shape (B, N)
            L_corr (CholeskyMatrix): Lower triangular Cholesky factor of the correlation matrix of shape (B, N, N)
        Returns:
            Tensor: Sampled vector z of shape (B, N)
        """

        eps = torch.randn_like(mu)
        z_corr = torch.einsum("bij,bj->bi", L_corr, eps)

        return mu + sigma * z_corr

    def kl_reg(self) -> Tensor:
        """
        Compute the KL divergence between N(mu, Sigma) and N(0, I)

        KL(N(mu, Sigma) || N(0, I)) = 0.5 * [ Tr(Sigma) + mu^T * mu - N - log|det(Sigma)| ]

        where:
        - Tr(Sigma) is the trace of the covariance matrix Sigma
        - mu^T * mu is the squared Mahalanobis distance
        - N is the dimensionality of the Gaussian
        - log|det(Sigma)| is the log-determinant of the covariance matrix Sigma
        """
        if self.mu is None or self.sigma is None or self.L_corr is None:
            raise RuntimeError(
                "Mean, Standard Deviation, and Cholesky factor must be computed before KL divergence calculation."
            )

        _, N = self.mu.shape

        # 1. Tr(Sigma) = sum(sigma^2)
        trace_sigma = torch.sum(torch.square(self.sigma), dim=1)

        # 2. mu^T @ mu
        mu_sq = torch.sum(torch.square(self.mu), dim=1)

        # 3. log|det(Sigma)| = 2*sum(log(sigma)) + 2*sum(log(diag(L_corr)))
        L_diag = torch.diagonal(self.L_corr, dim1=-2, dim2=-1)
        log_det_sigma = 2 * torch.sum(
            torch.log(self.sigma + 1e-8), dim=1
        ) + 2 * torch.sum(torch.log(L_diag + 1e-8), dim=1)

        kl = 0.5 * (trace_sigma + mu_sq - N - log_det_sigma)

        return kl.mean()  # Mean over the batch
