import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..typing import CholeskyMatrix
from .utils import cholesky


class Cholesky(nn.Linear):
    def __init__(
        self,
        in_features: int,
        n_variables: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        n_chol = n_variables * (n_variables + 1) // 2
        super().__init__(in_features, n_chol, bias, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> CholeskyMatrix:
        cholesky_params = F.linear(input, self.weight, self.bias)
        cholesky_params[:, self._diag_idx] = (  # Force positive diagonal
            torch.exp(cholesky_params[:, self._diag_idx]) + 1e-6
        )
        return cholesky(cholesky_params)

    @property
    def _diag_idx(self):
        i = torch.arange(self.n_variables, dtype=torch.int32)
        return (i * (i + 3)) // 2

    @property
    def n_variables(self) -> int:
        from math import sqrt

        n = (-1 + sqrt(1 + 8 * self.out_features)) / 2
        return int(n)
