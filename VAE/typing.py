from typing import Annotated, TypeAlias

from torch import Tensor

Correlation: TypeAlias = Annotated[
    Tensor,
    (
        "Correlation coefficients of shape (n_samples, n_corr) "
        "where n_corr = n_variables * (n_variables - 1) / 2"
    ),
]

CholeskyFactor: TypeAlias = Annotated[
    Tensor, "Cholesky factor of shape (B, C), C = N(N+1)/2"
]
CholeskyMatrix: TypeAlias = Annotated[
    Tensor, "lower triangular matrix with real and positive diagonal entries"
]

Mean: TypeAlias = Annotated[Tensor, "Mean vectors of shape (n_samples, n_variables)"]

StandardDeviation: TypeAlias = Annotated[
    Tensor, "Standard deviations of shape (n_samples, n_variables)"
]
