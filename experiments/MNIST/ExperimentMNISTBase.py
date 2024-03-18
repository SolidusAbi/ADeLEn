import config
import numpy as np
import os
import torch

from ADeLEn.model import ADeLEn
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from ..ExperimentBase import ExperimentBase


class ExperimentMNISTBase(ExperimentBase):
    def __init__(self, seed=42) -> None:
        self.model = ADeLEn((28, 28), [1, 12, 32], [1024, 256, 32], bottleneck=2)
        self.seed = seed
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

    
    def save_model(self):
        config = self.config()
        path = os.path.join(config['save_model_dir'], config['save_model_name'])
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        return self.model.load_model(path)
    
    
    def plot_reconstructed(self, model:ADeLEn, r0=(-6, 6), r1=(-6, 6), n=15):
        ''' 
            Plot the reconstructed images from the bottleneck.

            Args:
            -----
            model: ADeLEn
                The model to be used for reconstruction

            r0: tuple
                Range for the first dimension

            r1: tuple
                Range for the second dimension

            n: int
                Number of points to be sampled

            Returns:
            --------
            matplotlib.figure.Figure
                The figure with the reconstructed images
        '''
        
        w = 28
        img = np.zeros((n*w, n*w))

        fig = plt.figure(figsize=(5, 5))

        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]])
                x_hat = torch.tanh(model.decode_path(z)) # ADeLEn
                x_hat = x_hat.reshape(w, w).to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
        
        plt.xlabel('$\mathcal{N}(0, \sigma_1)$', fontsize='x-large')
        plt.ylabel('$\mathcal{N}(0, \sigma_2)$', fontsize='x-large')
        plt.tick_params(axis='both', which='major', labelsize='large')
        plt.imshow(img, extent=[*r0, *r1], cmap='viridis')

        return fig
    
    def to_test(self):
        '''
            Test the model with the test dataset

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
        x, y = zip(*[(_x, _y) for _x, _y in self.test_dataset])
        x = torch.stack(x)
        y = torch.tensor(y)

        _ = self.model(x)
        sigma = self.model.bottleneck.sigma
        y_score = torch.sum(sigma, dim=1).detach().numpy()
        
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)

        return (tpr, fpr, roc_auc)
    
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