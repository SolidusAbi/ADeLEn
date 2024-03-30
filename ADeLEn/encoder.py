import torch
from torch import nn
from abc import ABC, abstractmethod
from .utils import slide, SkipConnectionSequential

class Encoder(nn.Module, ABC):
    """
        Encoder class for downsampling the input tensor.

        Parameters:
        -----------
            channels (list): List of integers representing the number of input and output channels for each encoding module.
            skip_connection (bool): Flag indicating whether to use skip connections in the encoding path.

        Attributes:
        -----------
            encode_path (nn.Module): Sequential module containing the encoding modules.
    """
    def __init__(self, features:list, to_bottleneck=True, skip_connection=False) -> None:
        super().__init__()
        super(Encoder, self).__init__()

        encode_path = map(lambda x: self.__encode_module__(*x[1], dropout=True if not to_bottleneck or (x[0] < len(features)-2) else False), enumerate(slide(features, 2)))
        # encode_path = map(lambda x: self.__encode_module__(*x[1], dropout=True), enumerate(slide(features, 2)))
        self.encode_path = nn.Sequential(*encode_path) if not skip_connection else SkipConnectionSequential(*encode_path)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:          
        """
            Forward pass of the encoder.

            Parameters:
            -----------
                x (torch.Tensor): Input tensor.

            Returns:
            --------
                torch.Tensor: Output tensor after encoding.
        """
        return self.encode_path(x)
    
    @abstractmethod
    def get_encoded_size(self, input_size):
        pass
    
    @abstractmethod
    def __encode_module__(self, in_channels, out_channels, dropout=True):
        pass
        

class ConvEncoder(Encoder):
    def __init__(self, channels:list, to_bottleneck=True, skip_connection=False) -> None:
        """
            Encoder class for downsampling the input tensor.

            Parameters:
            -----------
                channels (list): List of integers representing the number of input and output channels for each encoding module.
                skip_connection (bool): Flag indicating whether to use skip connections in the encoding path.

            Attributes:
            -----------
                encode_path (nn.Module): Sequential module containing the encoding modules.
        """
        super(ConvEncoder, self).__init__(channels, to_bottleneck, skip_connection)

    def get_encoded_size(self, input_size:tuple) -> tuple:
        """ 
        Returns the size of the output tensor of the encoder.

        Parameters:
        -----------
            input_size (tuple): The size of the input tensor that corresponds to an image.
            The input_size can be (n_channels, height, width) or (height, width).

        Returns:
        --------
            tuple: The image dimension in the output tensor of the encoder.
        """
        assert isinstance(input_size, tuple), "input_size must be a tuple"

        if len(input_size) == 3:
            input_size = input_size[1:] 

        out_channels = self.encode_path[-1][0].out_channels 
        out_size = tuple(map(lambda x: x // (2**(len(self.encode_path))), input_size))
        return (out_channels, *out_size)
    
    def __encode_module__(self, in_channels, out_channels, dropout=True):
        '''
        The encoding path is designed to reduce the spatial dimensions of 
        the input tensor by a factor of 2. For example, if the input image
        is 240x240 pixels, the output image will be downscaled to 120x120 pixels.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            *(nn.Dropout2d(0.2), nn.ReLU()) if dropout else (nn.ReLU(),),   
            nn.MaxPool2d(3, stride=2, padding=1)
        )

class LinearEncoder(Encoder):
    def __init__(self, features:list, to_bottleneck=True, skip_connection=False) -> None:
        """
            Encoder class for downsampling the input tensor.

            Parameters:
            -----------
                features (list): List of integers representing the number of input and output features for each encoding module.
                skip_connection (bool): Flag indicating whether to use skip connections in the encoding path.

            Attributes:
            -----------
                encode_path (nn.Module): Sequential module containing the encoding modules.
        """
        super(LinearEncoder, self).__init__(features, to_bottleneck, skip_connection)

    def get_encoded_size(self, input_size:int) -> int:
        """ 
        Returns the number of elements in the output tensor of the encoder.

        Parameters:
        -----------
            input_size (int): The size of the input tensor.

        Returns:
        --------
            int: The number of elements in the output tensor of the encoder.
        """
        return self.encode_path[-1][0].out_features
    
    def __encode_module__(self, in_features, out_features, dropout=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=True),
            *(nn.Dropout(0.5), nn.ReLU()) if dropout else (nn.ReLU(),)
        )