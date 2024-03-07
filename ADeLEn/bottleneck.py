import torch
from torch import nn
from VAE.AnomalyDetector import AnomalyDetector 

from .utils import slide

class Bottleneck(nn.Module):
    def __init__(self, layers:list) -> None:
        super(Bottleneck, self).__init__()
        self.anomaly_detector = AnomalyDetector(layers[-2], layers[-1])

        self.bottleneck = nn.Sequential(
            nn.Sequential(
                *map(lambda x: self.__transform_module__(*x), slide(layers[:-1], 2))
            ),
            self.anomaly_detector,
            nn.Sequential(
                *map(lambda x: self.__transform_module__(*x), slide(layers[::-1], 2))
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x)

    def __transform_module__(self, in_features:int, out_features:int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
            nn.ReLU()
        )
