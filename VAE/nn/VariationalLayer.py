from abc import ABC, abstractmethod

from torch import Tensor


class VariationalLayer(ABC):
    """
    Abstract base class for variational models which is mandatory definces
    the KL divergence and the reparameterization trick methods.
    """

    @abstractmethod
    def _reparameterization(self, *args, **kwargs) -> Tensor: ...

    @abstractmethod
    def kl_reg(self, *args, **kwargs) -> Tensor: ...
