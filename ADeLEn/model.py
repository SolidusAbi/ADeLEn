import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import Bottleneck

class ADeLEn(nn.Module):
    '''
        Constraints:
            1. image size must be a square.
            2. Encoder's output estimated in the initialization.
    '''
    def __init__(self, img_size:int, encoder_channels:list, bottleneck_layers:list, skip_connection=False) -> None:
        super(ADeLEn, self).__init__()
        self.skip_connection = skip_connection

        self.encoder = Encoder(encoder_channels, skip_connection)
        encoded_size = self.encoder.get_encoded_size(img_size)
        out_img_size = self.encoder.get_encoded_image_size(img_size)
        
        bottleneck_layers.insert(0, encoded_size)
        self.bottleneck = nn.Sequential (
            nn.Flatten(),
            Bottleneck(bottleneck_layers),
            nn.Unflatten(1, (encoder_channels[-1], out_img_size, out_img_size))
        )
        self.decoder = Decoder(encoder_channels[::-1], skip_connection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sk = None 
        x = self.encoder(x)
        if self.skip_connection:
            x, sk = x            
        x = self.bottleneck(x)
        x = self.decoder(x, skip=sk)
        return x