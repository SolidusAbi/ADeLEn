import torch
from torch import nn
from abc import ABC, abstractmethod
from .utils import slide

class Decoder(nn.Module, ABC):
    def __init__(self, features:list, skip_connection=False, output_layer=False) -> None:
        super(Decoder, self).__init__()
        self.skip_connection = skip_connection

        self.decode_path = nn.Sequential(
            *map(
                lambda x: self.__decode_module__(
                    *x[1], # Unpack the tuple, in_channels and out_channels
                    activation=True if not output_layer or (x[0] < len(features)-2) else False,
                    skip_connection=skip_connection
                ),
                enumerate(slide(features, 2))
            )
        )


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.skip_connection:
            return self.__forward_skip_connection__(x, **kwargs)

        return self.decode_path(x)
    
    @abstractmethod
    def __forward_skip_connection__(self, x: torch.Tensor, skip:list) -> torch.Tensor:
        pass
    
    @abstractmethod
    def __decode_module__(self, in_channels, out_channels, activation=True, skip_connection=False):
        pass

class ConvDecoder(Decoder):
    def __init__(self, channels:list, skip_connection=False, output_layer=True) -> None:
        super(ConvDecoder, self).__init__(channels, skip_connection, output_layer)
        
    def __forward_skip_connection__(self, x: torch.Tensor, skip:list) -> torch.Tensor:
        assert len(skip) == len(self.decode_path), f'{len(skip)} != {len(self.decode_path)}'

        from torch.nn.functional import dropout2d
        for idx, layer in enumerate(self.decode_path[:-1]):
            up, *layers = layer
            x = up(x)
            x = torch.cat((x, dropout2d(skip[idx], p=.5, training=self.training)), dim=1) # Include dropout to the skip connection
            for layer in layers:
                x = layer(x)                   
        return self.decode_path[-1](x)
    
    def __decode_module__(self, in_channels, out_channels, activation=True, skip_connection=False):
        '''
            The decoding path is designed to increase the spatial dimensions of 
            the input tensor by a factor of 2. For example, if the input image
            is 120x120 pixels, the output image will be upscaled to 240x240 pixels.
        '''
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(2*in_channels if skip_connection and activation else in_channels, out_channels, 3, stride=1, padding=1),
            # *(nn.BatchNorm2d(out_channels, affine=True), nn.Dropout2d(0.2), nn.ReLU()) if activation else (nn.Identity(),)
            *(nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()) if activation else (nn.Identity(),)
        )
    
class LinearDecoder(Decoder):
    def __init__(self, in_features:list, skip_connection=False, output_layer=True) -> None:
        super(LinearDecoder, self).__init__(in_features, skip_connection, output_layer)  
    
    def __forward_skip_connection__(self, x: torch.Tensor, skip:list) -> torch.Tensor:
        assert len(skip) == len(self.decode_path)
        for idx, layer in enumerate(self.decode_path[:-1]):
            x = layer(x)
            x = torch.cat((x, skip[idx]), dim=1)
        return self.decode_path[-1](x)
    
    def __decode_module__(self, in_features, out_features, activation=True, skip_connection=False):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            # *(nn.BatchNorm1d(out_features, affine=True), nn.Dropout(0.2), nn.ReLU()) if activation else (nn.Identity(),)
            *(nn.BatchNorm1d(out_features, affine=False), nn.ReLU()) if activation else (nn.Identity(),)
        )