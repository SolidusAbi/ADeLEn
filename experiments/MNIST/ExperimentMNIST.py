
import torch

from dataset import AnomalyMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from .ExperimentMNISTBase import ExperimentMNISTBase
from ..utils import train

class ExperimentMNIST(ExperimentMNISTBase):
    '''
        Experiment class for MNIST with anomalies

        Args:
        -----
        anomalies: float
            Percentage of anomalies in the dataset

        pollution: float
            Percentage of pollution in the dataset

        seed: int
            Seed for reproducibility
    '''

    def __init__(self, anomalies, pollution, seed=None) -> None:
        self.anomalies_percent = anomalies
        self.experiment = 'mnist_anomalies_{}_pollution_{}'.format(anomalies, pollution)

        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.train_dataset = AnomalyMNIST('data/', download=True, transform=transform, n_normal_samples=2000, known_anomalies=anomalies, pollution=pollution, seed=seed)
        self.test_dataset = self.__prepare_test_dataset__('data/', transform, download=True)

        super().__init__(seed)
          
    def run(self) -> None:
        self.model = train(**self.config())

