import torch
from torch import nn
from .utils import slide, SkipConnectionSequential

class Encoder(nn.Module):
    def __init__(self, channels:list, skip_connection=False) -> None:
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
        super(Encoder, self).__init__()
       
        encode_path = map(lambda x: self.__encode_module__(*x), slide(channels, 2))
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
        out_img_size = self.get_encoded_image_size(input_size)
        out_channels = self.encode_path[-1][0].out_channels 
        return out_channels*(out_img_size**2)

    def get_encoded_image_size(self, input_size:int) -> int:
        """ 
            Returns the size of the output tensor of the encoder.

            Parameters:
            -----------
                input_size (int): The size of the input tensor.

            Returns:
            --------
                int: The size of the output tensor of the encoder.
        """
        return input_size // (2**(len(self.encode_path)))
    
    def __encode_module__(self, in_channels, out_channels):
        '''
        The encoding path is designed to reduce the spatial dimensions of 
        the input tensor by a factor of 2. For example, if the input image
        is 240x240 pixels, the output image will be downscaled to 120x120 pixels.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
