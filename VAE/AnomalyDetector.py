import torch

from .VariationalLayer import VariationalLayer
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn

class AnomalyDetector(nn.Linear, VariationalLayer):
    '''
        This implementation is for anomaly detection.
    '''
    def __init__(self, in_features, out_features, bias=True, sigma_anomaly=3) -> None:
        super(AnomalyDetector, self).__init__(in_features, out_features, bias)
        self.log_sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_bias = Parameter(torch.Tensor(out_features))
        torch.nn.init.xavier_uniform_(self.log_sigma_weight)
        self.log_sigma_bias.data.fill_(-5)
        self.mu, self.sigma = None, None

        self.sigma_anomaly = sigma_anomaly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = F.linear(x, self.weight, self.bias) 
        # sigma = torch.exp(F.linear(x, self.log_sigma_weight))
        # sigma = torch.exp(F.linear(x, self.log_sigma_weight, self.log_sigma_bias))
        # sigma = torch.exp(0.5 * F.linear(x*x, self.log_sigma_weight, self.log_sigma_bias))
        sigma = torch.exp(0.5 * F.linear(x, self.log_sigma_weight, self.log_sigma_bias))

        # if self.training:
        self.mu, self.sigma = mu, sigma

        # Reparameterization trick
        eps = torch.normal(0, torch.ones_like(sigma))
        return mu + sigma * eps

    def kl_reg(self, targets: torch.Tensor) -> torch.Tensor:
        # KL-Divergence regularization
        assert torch.all((targets == 0) | (targets == 1))
        sigma_2 = (torch.ones_like(targets) + ((self.sigma_anomaly-1) * targets)).unsqueeze(1).cuda()
        result = (torch.log(sigma_2) - torch.log(self.sigma) + (self.sigma**2 + self.mu**2)/(2*sigma_2**2) - 0.5)
        
        return result.mean() 
