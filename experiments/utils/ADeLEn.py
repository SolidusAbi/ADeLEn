import numpy as np
import pandas as pd
import torch


from itertools import chain
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from VAE.loss import SGVBL


from .misc import weights

def train(model, train_dataset, batch_size, n_epochs, lr=1e-3, kl_weight=1, weighted_sampler=False, **kwargs):
    if weighted_sampler:
        sampler = torch.utils.data.WeightedRandomSampler(weights(train_dataset), len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    epoch_iterator = tqdm(
            range(n_epochs),
            leave=False,
            unit="epoch",
            postfix={"tls": "%.4f" % -1},
        )
    
    opt = Adam(model.parameters(), lr=lr)
    sgvbl = SGVBL(model, len(train_dataset), mle=mse_loss)
    for _ in epoch_iterator:
        epoch_loss = 0.
        for x, y in train_loader:
            x = x.to(device) 
            opt.zero_grad()
            x_hat = torch.tanh(model(x))
            loss = sgvbl(x, x_hat, y, kl_weight)
            epoch_loss += loss.detach().item()

            loss.backward()
            opt.step()

        epoch_iterator.set_postfix(tls="%.3f" % (epoch_loss/len(train_loader)))

    return model.eval().cpu()

# def train(model, train_dataset, batch_size, n_epochs, lr=1e-3, kl_weight=1, **kwargs):
#     from tqdm import tqdm
#     from torch.utils.data import DataLoader
#     from torch.optim import Adam
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.train()

#     epoch_iterator = tqdm(
#             range(n_epochs),
#             leave=False,
#             unit="epoch",
#             postfix={"tls": "%.4f" % -1},
#         )
    
#     opt = Adam(model.parameters(), lr=lr)
#     sgvbl = SGVBL(model, len(train_dataset), mle=mse_loss)
#     for _ in epoch_iterator:
#         epoch_loss = 0.
#         for x, y in train_loader:
#             x = x.to(device) 
#             opt.zero_grad()
#             x_hat = torch.tanh(model(x))
#             loss = sgvbl(x, x_hat, y, kl_weight)
#             epoch_loss += loss.detach().item()

#             loss.backward()
#             opt.step()

#         epoch_iterator.set_postfix(tls="%.3f" % (epoch_loss/len(train_loader)))

#     return model.eval().cpu()


def generate_roc_df(roc_list:list) -> pd.DataFrame:
    ''' 
        Create a DataFrame from a list of roc curves
        Args:
        -----
            roc_list: list
                List of N roc curves where N is the number of iterations. It is a 
                list of tuples of the form (fpr, tpr), where fpr is the false positive
                rate and tpr is the true positive rate.
        Returns:
        --------
            roc_df: pd.DataFrame
                DataFrame with multiindex
    '''
    index_names = [
        list(map(lambda x: 'It {}'.format(x), np.repeat(np.arange(len(roc_list)), 2) + 1 )),
        ['FPR', 'TPR']*len(roc_list)
    ]
        
    tuples = list(zip(*index_names))
    index = pd.MultiIndex.from_tuples(tuples)
    roc_df = pd.DataFrame(chain.from_iterable(roc_list), index=index)
    return roc_df