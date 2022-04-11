import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Module
from models.stylegan2.model import EqualLinear
from models.encoders.attention  import TransformerLayer, get_positional_encoding

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, attention=False):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        for i in range(num_pools - 4):
            modules += [Conv2d(in_c if i == 0 else out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs_large = nn.Sequential(*modules)

        modules = []
        for i in range(num_pools - 4, num_pools):
            modules += [Conv2d(in_c if i == 0 else out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs_small = nn.Sequential(*modules)

        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

        if attention:
            # Expected input sizes for attention layer
            in_res = 16
            in_channels = 544

            grid_pos, _ = get_positional_encoding(in_res, in_channels)
            self.register_buffer("grid_pos", grid_pos)

            self.transformer = TransformerLayer(
                dim = in_channels,
                pos_dim = in_channels,
                from_len = in_res * in_res,
                to_len = in_res * in_res,
                from_dim = in_channels,
                to_dim = in_channels
            )

    def forward(self, x, c=None):
        x = self.convs_large(x)
        if self.transformer is not None:
            assert c is not None and c.shape == x.shape            
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1)
            c = c.reshape(shape[0], shape[1], -1).permute(0, 2, 1)

            x, _, _ = self.transformer(
                from_tensor = x, to_tensor = c, 
                from_pos = self.grid_pos,   to_pos = None,
                hw_shape = shape[-2:]
            )
            x = x.permute(0, 2, 1).reshape(shape)
            
        x = self.convs_small(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x