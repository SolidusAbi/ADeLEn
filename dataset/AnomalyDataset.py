from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class AnomalyDataset(Dataset):
    def __init__(self, dataset, n_normal_samples=-1, known_anomalies=0.2, pollution=0.1, seed=None) -> None:
        super().__init__()

        self.mnist = dataset
        rng = torch.Generator() if not seed else torch.Generator().manual_seed(seed)
        normal_idx, anomaly_idx = self.create_subset(dataset, rng, n_normal_samples, known_anomalies, pollution)
        
        dataset.targets = -1 * np.ones(len(dataset.targets))
        dataset.targets[normal_idx] = 0
        dataset.targets[anomaly_idx] = 1

        self.subset = Subset(dataset, torch.cat([normal_idx, anomaly_idx]))
        self.n_pollution = int((len(normal_idx)/(1+pollution))*pollution)
        self.n_known_anomalies = len(anomaly_idx)

    def create_subset(self, dataset:Dataset, rng:torch.Generator, samples=-1, known_anomalies=0.2, pollution=0.05) -> tuple:
        '''
            Create a subset of the dataset for anomaly detection. The indices of the normal
            samples are contaimnated by the 5% of unknown anomalies.

            Args:
                dataset: Dataset
                    Original dataset.
                
                rng: torch.Generator
                    Random number generator to use for the random permutation of the anomaly indices.

                samples: int
                    Number of normal samples to use in the dataset. If -1, all the normal samples are used.

                known_anomalies: float
                    Percentage of known anomalies to use in the dataset.
                    
                pollution: float
                    Percentage of unknown anomalies to include in the dataset.
                

            Returns:
            --------
                normal_idx: list
                    List of indices of the normal samples, including the unknown anomalies.
                anomaly_idx: list
                    List of indices of the anomaly samples.
        '''
        normal_idx = torch.tensor(np.where((dataset.targets == 0))[0])
        anomaly_idx = torch.tensor(np.where((dataset.targets == 1))[0])
        
        normal_rnd_perm = torch.randperm(len(normal_idx), generator=rng) 
        normal_idx = normal_idx[normal_rnd_perm[:samples]] if samples != -1 else normal_idx
        
        n_unknown_anomalies, n_known_anomalies =  int(len(normal_idx) * pollution), int(len(normal_idx) * known_anomalies)
        anomaly_rnd_perm = torch.randperm(len(anomaly_idx), generator=rng)
        anomaly_idx = anomaly_idx[anomaly_rnd_perm[:n_unknown_anomalies + n_known_anomalies]]
        
        normal_idx = torch.cat([normal_idx, anomaly_idx[:n_unknown_anomalies]]) # include the unknown anomalies in the normal samples
        anomaly_idx = anomaly_idx[n_unknown_anomalies:n_unknown_anomalies+n_known_anomalies]

        return (normal_idx, anomaly_idx)
    
    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx) -> tuple:
        return self.subset[idx]
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} Dataset (Number of samples: {len(self.subset)}, "
            f"Number of known anomalies: {self.n_known_anomalies}, "
            f"Number of unknown anomalies: {self.n_pollution})")
    
    def montage(self, n_row = 5, n_col= 5, seed=None):
        '''
            Plot a montage of the dataset.
            
            Args:
                n_row: int, optional
                    Number of rows of the montage.
                n_col: int, optional
                    Number of columns of the montage.
                seed: int, optional
                    Seed to use for the random permutation of the anomaly indices.

            Returns:
                fig: matplotlib.pyplot.figure
                    Figure of the montage.
        '''
        rng = torch.Generator() if not seed else torch.Generator().manual_seed(seed)
        rng = torch.randperm(len(self.subset), generator=rng)
        fig = plt.figure(figsize=(10,10))
        length = n_row*n_col

        plt.subplots_adjust(hspace=0.5)
        for i in range(length):
            plt.subplot(n_row,n_col,i+1)
            x, y = self.subset[rng[i]]
            plt.imshow(x.squeeze(), cmap='gray')
           
            if y == 1:
                plt.gca().spines['top'].set_color('red')
                plt.gca().spines['bottom'].set_color('red')
                plt.gca().spines['left'].set_color('red')
                plt.gca().spines['right'].set_color('red')

                plt.gca().spines['top'].set_linewidth(5)
                plt.gca().spines['bottom'].set_linewidth(5)
                plt.gca().spines['left'].set_linewidth(5)
                plt.gca().spines['right'].set_linewidth(5)

                plt.xticks([]), plt.yticks([])
            else:
                 plt.axis('off')

        return fig