import config
import numpy as np
import os
import torch

from dataset import AnomalyMNIST
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ..ExperimentBase import ExperimentBase


class ExperimentMNISTBase(ExperimentBase):
    def __init__(self, known_anomalies, pollution, seed=42) -> None:
        self.seed = seed
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.train_dataset = AnomalyMNIST('data/', download=True, transform=transform, n_normal_samples=2000, known_anomalies=known_anomalies, pollution=pollution, seed=seed)
        self.test_dataset = self.__prepare_test_dataset__('data/', transform, download=True)
        super().__init__()


    def config(self) -> dict:
        result_dir = os.path.join(config.RESULTS_DIR, 'Chapter8/MNIST/{}'.format(self.experiment))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, )
            os.makedirs(os.path.join(result_dir, 'model'))
            os.makedirs(os.path.join(result_dir, 'imgs'))
            os.makedirs(os.path.join(result_dir, 'result'))

        return {
            'experiment_name': self.experiment,
            'dataset_dir': './data/',
            'model': self.model,
            'train_dataset': self.train_dataset,
            'batch_size': 128,
            'n_epochs': 50,
            'lr': 1e-3,
            'kl_weight': 1,
            'seed': self.seed,
            'save_dir': result_dir,
            'save_result': True,
            'save_result_dir': os.path.join(result_dir, 'result'),
            'save_imgs_dir': os.path.join(result_dir, 'imgs'),
            'save_model_dir': os.path.join(result_dir, 'model'),
            'save_model_name': 'model.pt',
        }
    

    def save_config(self) -> None:
        config = self.config()
        with open(os.path.join(config['save_dir'], 'config.txt'), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}: {v}\n')

    
    def save_model(self, verbose=True):
        config = self.config()
        path = os.path.join(config['save_model_dir'], config['save_model_name'])
        self.model.save(path)
        if verbose:
            print(f"Model saved to {path}")


    def load_model(self, path):
        return self.model.load_model(path)
    
    
    def __prepare_test_dataset__(self, root, transform, download=True) -> Subset:
        ''' 
            Prepare the test dataset

            Args:
            -----
            root: str
                Root directory where the dataset is stored

            transform: torchvision.transforms
                Transformations to be applied to the dataset

            download: bool
                Whether to download the dataset or not

            Returns:
            --------    
            Subset
                The test dataset
        '''
        dataset = MNIST(root, train=False, transform=transform, download=download)
        normal_idx = torch.where((dataset.targets == 1))[0]
        anomaly_idx = torch.where((dataset.targets == 7))[0] 
        idx = torch.cat([normal_idx, anomaly_idx])

        dataset.targets = torch.ones_like(dataset.targets) * -1
        dataset.targets[normal_idx] = 0
        dataset.targets[anomaly_idx] = 1

        return Subset(dataset, idx)