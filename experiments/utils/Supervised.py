import torch

from functools import reduce
from torch import nn
from ADeLEn.encoder import ConvEncoder, LinearEncoder

from .misc import weights

class SupervisedModel(nn.Module):
    def __init__(self, img_size:int, conv_layers:list, linear_layers:list) -> None:
        super(SupervisedModel, self).__init__()
        self.model = self.__model__(conv_layers, linear_layers, img_size)

    def __model__(self, channels:list, linear_layers:list, img_size):
        conv_encoder = ConvEncoder(channels, to_bottleneck=False)
        _out_size = reduce(lambda x, y: x*y, conv_encoder.get_encoded_size(img_size))
        linear_encoder = LinearEncoder([_out_size, *linear_layers], to_bottleneck=True)

        return nn.Sequential(
            conv_encoder,
            nn.Flatten(),
            linear_encoder,
            nn.Linear(linear_layers[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()
    
    def save(self, path:str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path:str) -> None:
        self.load_state_dict(torch.load(path))

    def score_samples(self, X:torch.Tensor) -> torch.Tensor:
        ''' 
            Compute the score for the samples. In this case, it is considered
            the ouput of 'anomaly' neuron in the last layer of the model.

            Args:
            -----
            X: torch.Tensor
                The samples to be scored.

            Returns:
            --------
                score: torch.Tensor
                    The score for the samples.
        '''
        # with torch.no_grad():
        #     y_hat = torch.softmax(self.model(X), dim=1)
        # return y_hat[:, 1]
        with torch.no_grad():
            return torch.sigmoid(self.model(X).flatten())
    

def train(model, train_dataset, batch_size, n_epochs, lr=1e-3, weighted_sampler=False, **kwargs):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn.functional import one_hot
    
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
    # ce = nn.CrossEntropyLoss()
    ce = nn.BCEWithLogitsLoss()
    for _ in epoch_iterator:
        epoch_loss = 0.
        for x, y in train_loader:            
            y = y.to(device).float()
            x = x.to(device) 
            opt.zero_grad()
            y_hat = model(x)
            loss = ce(y_hat, y)
            epoch_loss += loss.detach().item()

            loss.backward()
            opt.step()

        epoch_iterator.set_postfix(tls="%.3f" % (epoch_loss/len(train_loader)))

    return model.eval().cpu()