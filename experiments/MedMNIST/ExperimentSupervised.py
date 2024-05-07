import numpy as np
import os
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .ExperimentMedMNISTBase import ExperimentMedMNISTBase
from ..utils.Supervised import SupervisedModel, train

class ExperimentSupervised(ExperimentMedMNISTBase):
    def __init__(self, known_anomalies, pollution, seed=None, **kwargs) -> None:
        self.model = SupervisedModel((28, 28), [1, 16, 24], [1024, 256, 32])
        self.anomalies_percent = known_anomalies
        self.experiment = 'Supervised/medmnist_anomalies_{}_pollution_{}'.format(known_anomalies, pollution)

        super().__init__(known_anomalies, pollution, seed)

    def config(self) -> dict:
        base_config = super().config()
        base_config.update({
            'batch_size': 128,
            'n_epochs': 50,
            'lr': 1e-3,
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
        config = self.config()
        path = os.path.join(config['save_model_dir'], config['save_model_name'])
        return self.model.load_model(path)
    
    def classification_metrics(self) -> tuple:
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
        y_pred = np.where(scores > .5, 1, 0)
        
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
        y = torch.tensor(np.array(y).flatten())
        return (X, y)