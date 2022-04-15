import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Module
from models.stylegan2.model import EqualLinear
from models.encoders.attention  import TransformerLayer, get_positional_encoding

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, attention=False, context_res=None, context_dim=None):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

        self.transformer = None
        if attention:
            assert context_res is not None and context_dim is not None
            # Expected input sizes for attention layer
            self.style_len = 16
            self.style_dim = 32

            grid_pos, _ = get_positional_encoding(context_res, pos_dim=self.style_dim)
            self.register_buffer("grid_pos", grid_pos)

            self.transformer = TransformerLayer(
                dim = self.style_dim,
                pos_dim = self.style_dim,
                from_len = self.style_len,
                to_len = context_res * context_res,
                from_dim = self.style_dim,
                to_dim = context_dim
            )

    def forward(self, x, c=None):
        x = self.convs(x)

        if self.transformer is not None:
            x_components, x_global = x[:, :-self.style_dim], x[:, -self.style_dim:] 
                 
            x_components = x_components.reshape(-1, self.style_len, self.style_dim)
            c = c.reshape(c.shape[0], c.shape[1], -1).permute(0, 2, 1)

            x_components, _, _ = self.transformer(
                from_tensor = x_components, to_tensor = c, 
                from_pos = None,   to_pos = self.grid_pos,
                hw_shape = (4, 4)
            )

            x_components = x_components.reshape(x_components.shape[0], -1, 1, 1)
            x = torch.cat((x_components, x_global), dim=1)
        
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
