import torch

from ..typing import CholeskyFactor, CholeskyMatrix


def cholesky(params: CholeskyFactor) -> CholeskyMatrix:
    B, P = params.shape
    N = int(((8 * P + 1) ** 0.5 - 1) / 2)  # Solve N(N+1)/2 = P  => N^2 + N - 2*P = 0

    L = torch.zeros((B, N, N), device=params.device, dtype=params.dtype)
    tril_indices = torch.tril_indices(N, N, device=params.device)
    L[:, tril_indices[0], tril_indices[1]] = params

    return L


def correlation(params: CholeskyFactor) -> CholeskyMatrix:
    L = cholesky(params)
    return L / torch.norm(L, p=2, dim=-1, keepdim=True)
