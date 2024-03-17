from .AnomalyDataset import AnomalyDataset
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class AnomalyMNIST(AnomalyDataset):
    ''' 
        Anomaly detection dataset for MNIST. The dataset is composed by the normal samples (1) and
        the anomaly samples (7). The known anomalies are anotated by the label 1, while the normal and
        `unknown' anomalies samples are anotated by the label 0.

        Args:
            root: str
                Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            download: bool, optional
                If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

            transform: callable, optional
                A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``. By default,
                ``torchvision.transforms.ToTensor`` is applied.
            
            n_normal_samples: int, optional
                Number of normal samples to use in the dataset.
                
            known_anomalies: float, optional
                Percentage of known anomalies to use in the dataset. The number of known anomalies is 
                computed based on the number of n_normal_samples value.

            pollution: float, optional
                Percentage of unknown anomalies to include in the dataset. The percentage is
                calculated based on the number of n_normal_samples value.

            seed: int, optional
                Seed to use for the random permutation of the anomaly indices.
    '''
              
    def __init__(self, root, download=True, transform = ToTensor(), n_normal_samples=-1,
                  known_anomalies=0.2, pollution=0.1, seed=None)->None:

        mnist = MNIST(root = root, train=True, download=download, transform = transform)
        normal_idx = torch.where((mnist.targets == 1))[0]
        anomaly_idx = torch.where((mnist.targets == 7))[0]

        mnist.targets = -1 * torch.ones(len(mnist.targets))
        mnist.targets[normal_idx] = 0
        mnist.targets[anomaly_idx] = 1

        super().__init__(mnist, n_normal_samples, known_anomalies, pollution, seed)
