
import numpy as np
import os
import torch


from ADeLEn.model import ADeLEn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .ExperimentMNISTBase import ExperimentMNISTBase
from ..utils.ADeLEn import train


class ExperimentADeLEn(ExperimentMNISTBase):
    '''
        Experiment class for MNIST with anomalies

        Args:
        -----
            anomalies: float
                Percentage of known anomalies in the dataset
            pollution: float
                Percentage of pollution in the dataset
            d: int
                Dimension of the bottleneck
            seed: int
                Seed for reproducibility
    '''

    def __init__(self, known_anomalies, pollution, d=2, seed=None) -> None:
        self.model = ADeLEn((28, 28), [1, 32, 48], [1024, 256, 32], bottleneck=d)
        self.anomalies_percent = known_anomalies
        self.experiment = 'ADeLEn/mnist_anomalies_{}_pollution_{}_bottleneck_{}'.format(known_anomalies, pollution, d)

        super().__init__(known_anomalies, pollution, seed)
          
    def config(self) -> dict:
        base_config = super().config()
        base_config.update({
            'batch_size': 128,
            'n_epochs': 50,
            'lr': 1e-3,
            'kl_weight': 1,
            'weighted_sampler': False,
            'save_model_name': 'model.pt'
        })

        return base_config

    def run(self) -> None:
        self.model = train(**self.config())

        _, _, auc = self.roc_curve()
        return auc
    
    def save_model(self, verbose=True):
        config = self.config()
        path = os.path.join(config['save_model_dir'], config['save_model_name'])
        self.model.save(path)
        if verbose:
            print(f"Model saved to {path}")

    def load_model(self, path):
        return self.model.load_model(path)

    def plot_reconstructed(self, model:ADeLEn, r0=(-6, 6), r1=(-6, 6), n=15):
        ''' 
            Plot the reconstructed images from the bottleneck. It only works with 
            2-dimensoinal bottlenecks.

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
        if model.bottleneck.out_features != 2:
            raise ValueError("The bottleneck dimension must be 2")
        
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
        X, y = self.__get_test_data__()

        scores = self.model.score_samples(X)               
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        return (fpr, tpr, roc_auc)
    
    def classification_metrics(self, sigma=1.2) -> tuple:
        '''
            Test the model with the test dataset

            Returns:
            --------
                accuracy: float
                    The accuracy of the model.
                precision: float
                    The precision of the model.
                recall: float
                    The recall of the model.
                f1: float
                    The f1 score of the model.
        '''
        y, scores = self.score_samples()
        y_pred = np.where(scores > self._theshold(sigma), 1, 0)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return (accuracy, precision, recall, f1)
    
    def score_per_label(self) -> float:
        '''
            Test the model with the test dataset

            Returns:
            --------
                normal_scores: np.array
                    The scores of the normal samples
                anomaly_scores: np.array
                    The scores of the anomaly samples
        '''
        y, scores = self.score_samples()
        return (scores[y == 0], scores[y == 1])

    def _theshold(self, sigma)->float:
        d = self.model.bottleneck.out_features
        score = d * np.log(sigma)
        gauss = d * np.log(2*torch.pi*torch.e)
        return .5 * (gauss + score)
    
    def __get_test_data__(self) -> tuple:
        '''
            Get the test data

            Returns:
            --------
                X: np.array
                    The test data
                y: np.array
                    The labels
        '''
        X, y = zip(*self.test_dataset)
        X = torch.stack(X)
        y = torch.tensor(y)
        return (X, y)