
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import OneClassSVM

from .ExperimentMNISTBase import ExperimentMNISTBase

class ExperimentSVM(ExperimentMNISTBase):
    '''
        Experiment class for MNIST using One-Class SVM

        Args:
        -----
            pollution: float
                Percentage of pollution in the dataset

            seed: int
                Seed for reproducibility
    '''
    def __init__(self, known_anomalies, pollution, seed=None, **kwargs) -> None:
        self.model = OneClassSVM()
        self.anomalies_percent = 0
        self.experiment = f'SVM/mnist_pollution_{pollution}'

        super().__init__(known_anomalies, pollution, seed)


    def run(self, verbose=0) -> float:
        '''
            Run the experiment

            Args:
            -----
                verbose: int
                    Verbosity level used in the optimization process (GridSearchCV).

            Returns:
            --------
                auc: float
                    The AUC score of the model.
        '''
        best_params = self.__optimize_svm__(verbose=verbose)
        self.model = OneClassSVM(**best_params)
        x_train, _ = zip(* self.train_dataset)
        x_train = np.stack(x_train).reshape(-1, 28*28)
        self.model.fit(x_train)

        _, _, auc = self.roc_curve()
        return auc
    
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
        

        X, y = self.__get_test_data__()
        y_pred = self.model.predict(X)
        
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
        return (scores[y == 1], scores[y == -1])


    def __optimize_svm__(self, verbose) -> dict:
        '''
            Optimize the SVM model using an unfair advantage of selecting its hyperparameters optimally to maximize AUC
            on a subset (10%) of the test set.

            Args:
            -----
                verbose: int
                    Verbosity level used in the optimization process (GridSearchCV).
            
            Returns:
            --------
                best_params: dict
                    The best hyperparameters for the SVM model
        '''
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import PredefinedSplit

        x_train, y_train = zip(*self.train_dataset)
        x_train, y_train = np.stack(x_train), np.array(y_train)

        # # Using the known anomalies to optimize the SVM model
        # X = x_train.reshape(-1, 28*28)
        # y = y_train
        # test_fold = np.array([-1]*len(x_train))      
        # known_anomalies_idx = np.argwhere(y_train == 1).flatten()
        # test_fold[known_anomalies_idx] = 0
        # normal_idx = np.argwhere(y_train == 0).flatten()
        # idx = np.random.permutation(normal_idx)[:len(known_anomalies_idx)]
        # test_fold[idx] = 0

        # Using the 10 percent of the test set to optimize the SVM model
        x_test, y_test = self.__get_random_test_subset__(percent=.1)

        X = np.concatenate([x_train, x_test], axis=0).reshape(-1, 28*28)
        y = np.concatenate([y_train, y_test], axis=0)
        test_fold = [-1]*len(x_train) + [0]*len(x_test)

        
        ps = PredefinedSplit(test_fold)
        y = self.__change_labels__(y)

        param_grid = {'kernel': ['rbf'], 'gamma': np.logspace(-3, 0, 8),
                     'nu': [.01, .05, .1, .2, .5]}
        grid = GridSearchCV(OneClassSVM(), param_grid, scoring='roc_auc', verbose=verbose, cv=ps, n_jobs=-1)
        grid.fit(X, y)
        return grid.best_params_


    def __get_random_test_subset__(self, percent=.1):
        x_test, y_test = zip(*self.test_dataset)
        x_test, y_test = np.stack(x_test), np.array(y_test)
        n_samples = int(len(y_test) * percent)

        # Select the 10% subset in the test set
        np.random.seed(self.seed)
        idx = np.random.permutation(len(y_test))[:n_samples] 
        x = x_test[idx]
        y = y_test[idx]

        return (x, y)


    def __change_labels__(self, y:np.array):
        '''
            Change the labels from [0, 1] to [-1, 1]

            Args:
            -----
            y: np.array
                The labels to be changed

            Returns:
            --------
            np.array
                The changed labels
        '''
        return np.where(y == 0, 1, -1)

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
        X = np.stack(X).reshape(-1, 28*28)
        y = self.__change_labels__(np.array(y))
        return (X, y)