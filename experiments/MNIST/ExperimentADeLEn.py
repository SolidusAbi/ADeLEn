
import numpy as np
import torch

from ADeLEn.model import ADeLEn
from dataset import AnomalyMNIST
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize

from .ExperimentMNISTBase import ExperimentMNISTBase
from ..utils import train


class ExperimentADeLEn(ExperimentMNISTBase):
    '''
        Experiment class for MNIST with anomalies

        Args:
        -----
        anomalies: float
            Percentage of known anomalies in the dataset

        pollution: float
            Percentage of pollution in the dataset

        seed: int
            Seed for reproducibility
    '''

    def __init__(self, anomalies, pollution, seed=None) -> None:
        self.model = ADeLEn((28, 28), [1, 32, 48], [1024, 256, 32], bottleneck=10)
        self.anomalies_percent = anomalies
        self.experiment = 'ADeLEn/mnist_anomalies_{}_pollution_{}'.format(anomalies, pollution)

        super().__init__(anomalies, pollution, seed)
          
    def run(self) -> None:
        self.model = train(**self.config())


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