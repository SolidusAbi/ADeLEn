import config
import numpy as np
import os
import torch

from abc import abstractmethod
from dataset import AnomalyMNIST
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
            'seed': self.seed,
            'save_dir': result_dir,
            'save_result_dir': os.path.join(result_dir, 'result'),
            'save_imgs_dir': os.path.join(result_dir, 'imgs'),
            'save_model_dir': os.path.join(result_dir, 'model'),
        }
    
    def save_config(self) -> None:
        config = self.config()
        with open(os.path.join(config['save_dir'], 'config.txt'), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}: {v}\n')
    
    # def classification_metrics(self, **kwargs) -> tuple:


    def roc_curve(self):
        '''
            Obtain the ROC curve of the model with the test dataset.

            Returns:
            --------
                tpr: float
                    True positive rate.
                fpr: float
                    False positive rate.
                roc_auc: float
                    Area under the curve.
        '''
        from sklearn.metrics import roc_curve, auc
        y, scores = self.score_samples()
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        return (fpr, tpr, roc_auc)

    def score_samples(self) -> float:
        '''
            Obtain the scores of the samples and the labels.

            Returns:
            --------
                labels: 
                    label of the samples
                scores: np.array
                    The scores of the samples by the model
        '''
        X, y = self.__get_test_data__()
        scores = self.model.score_samples(X)
        return (y, scores)
    
    @abstractmethod
    def classification_metrics(self, **kwargs) -> tuple:
        pass
    
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