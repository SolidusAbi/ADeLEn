import numpy as np
import torch

from functools import reduce
from torch import nn

from .encoder import ConvEncoder, LinearEncoder
from .decoder import ConvDecoder, LinearDecoder
from .bottleneck import Bottleneck
from VAE.nn import AnomalyDetector


class ADeLEn(nn.Module):
    '''
        Constraints:
            1. image size must be a square.
            2. Encoder's output estimated in the initialization.
    '''
    def __init__(self, img_size:int, conv_encoder:list, linear_encoder:list, bottleneck=2, skip_connection=False) -> None:
        ''' 
            Skip Connection is not implemented yet.
        '''
        super(ADeLEn, self).__init__()
        self.skip_connection = skip_connection

        self.encode_path = self.__encode_path__(conv_encoder, linear_encoder, img_size)
        self.bottleneck = AnomalyDetector(linear_encoder[-1], bottleneck)
        self.decode_path = self.__decode_path__(conv_encoder[::-1], [bottleneck, *linear_encoder[::-1]], self.encode_path[0].get_encoded_size(img_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_path(x)
        # sk = None 
        # if self.skip_connection:
        #     x, sk = x            
        x = self.bottleneck(x)
        # x = self.decode_path(x, skip=sk)
        x = self.decode_path(x)
        return x
    
    def save(self, path:str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path:str) -> None:
        self.load_state_dict(torch.load(path))

    def score_samples(self, x:torch.Tensor, normalize=True) -> torch.Tensor:
        ''' 
            Compute the score for the samples. The score is the entropy of the 
            latent space Z.

            Args:
            -----
            x: torch.Tensor
                The samples to be scored.

            normalize: bool
                If True, the score is normalized by the number of dimensions.
        '''
        with torch.no_grad():
            _ = self.bottleneck(self.encode_path(x.detach()))
            score = torch.log(self.bottleneck.sigma.detach()).sum(dim=1).cpu().numpy()
        if normalize:
            _, d = self.bottleneck.sigma.shape
            gauss = d * np.log(2*torch.pi*torch.e)
            score = .5 * (gauss + score)
        return score

    def __encode_path__(self, channels:list, linear_encoder:list, img_size):
        conv_encoder = ConvEncoder(channels, to_bottleneck=False)
        _out_size = reduce(lambda x, y: x*y, conv_encoder.get_encoded_size(img_size))
        linear_encoder = LinearEncoder([_out_size, *linear_encoder], to_bottleneck=True)

        return nn.Sequential(
            conv_encoder,
            nn.Flatten(),
            linear_encoder
        )
    
    def __decode_path__(self, conv_decoder:list, linear_decoder:list, encoded_size:tuple):
        _out_size = reduce(lambda x, y: x*y, encoded_size)
        linear_decoder = LinearDecoder([*linear_decoder, _out_size], False, False)
        conv_decoder = ConvDecoder(conv_decoder, False, True)

        return nn.Sequential(
            linear_decoder,
            nn.Unflatten(1, encoded_size),
            conv_decoder
        )

# Old Implementation
# class ADeLEn(nn.Module):
#     '''
#         Constraints:
#             1. image size must be a square.
#             2. Encoder's output estimated in the initialization.
#     '''
#     def __init__(self, img_size:int, encoder_channels:list, bottleneck_layers:list, skip_connection=False) -> None:
#         super(ADeLEn, self).__init__()
#         self.skip_connection = skip_connection

#         self.encoder = ConvEncoder(encoder_channels, skip_connection)
#         encoded_size = self.encoder.get_encoded_size(img_size)
#         # out_img_size = self.encoder.get_encoded_image_size(img_size)
        
#         bottleneck_layers.insert(0, reduce(lambda x, y: x*y, encoded_size))
#         self.bottleneck = nn.Sequential (
#             nn.Flatten(),
#             Bottleneck(bottleneck_layers),
#             nn.Unflatten(1, encoded_size)
#         )
#         self.decoder = ConvDecoder(encoder_channels[::-1], skip_connection)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         sk = None 
#         x = self.encoder(x)
#         if self.skip_connection:
#             x, sk = x            
#         x = self.bottleneck(x)
#         x = self.decoder(x, skip=sk)
#         return x