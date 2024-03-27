import torch
import numpy as np

def weights(dataset):
    _, y = zip(*dataset)
    y = torch.tensor(y)

    count = torch.bincount(y)
    weights = 1. / np.array(count)
    weights /= weights.sum()

    return weights[y]