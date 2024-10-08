from medmnist.dataset import PneumoniaMNIST

from .AnomalyDataset import AnomalyDataset

class AnomalyPneumoniaMNIST(AnomalyDataset):
    def __init__(self, root, download=True, transform=None, n_normal_samples=-1, known_anomalies=0.2, pollution=0, seed=None) -> None:
        pneumonia = PneumoniaMNIST(split='train', transform=transform, download=download, root=root)
        pneumonia.targets = pneumonia.labels # It is just for standardization, the targets are the labels in any dataset
        super().__init__(pneumonia, n_normal_samples, known_anomalies, pollution, seed)
        pneumonia.labels = pneumonia.targets