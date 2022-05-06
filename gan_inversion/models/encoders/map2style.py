import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Module
from models.stylegan2.model import EqualLinear
from models.encoders.attention  import TransformerLayer, get_positional_encoding

class GansformerStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, context_res=None, context_dim=None, **_kwargs):
        super(GansformerStyleBlock, self).__init__()
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
        """
        Runs local components of the style through a final attention layer, 
        which is computed over those components using the context feature map.
        """
        x = self.convs(x)

        x_components, x_global = x[:, :-self.style_dim], x[:, -self.style_dim:] 
                
        x_components = x_components.reshape(-1, self.style_len, self.style_dim)
        c = c.reshape(c.shape[0], c.shape[1], -1).permute(0, 2, 1)

        x_components, att_map, _ = self.transformer(
            from_tensor = x_components, to_tensor = c, 
            from_pos = None,   to_pos = self.grid_pos,
            hw_shape = (4, 4)
        )

        x_components = x_components.reshape(x_components.shape[0], -1, 1, 1)
        x = torch.cat((x_components, x_global), dim=1)
        
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x, att_map

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
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

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class AttentionBlock(Module):
    """
    Block of attention layers used in StyleGAN2-based model
    """
    def __init__(self, input_res, style_dim, n_styles):
        super(AttentionBlock, self).__init__()
        self.style_dim = style_dim
        self.n_styles = n_styles
        transformer_modules = []
        start_channels = 64
        for i in range(5):
            # context resolutions are 16x16 -> 256x256 in powers of 2
            context_res = int(input_res / (2**(4-i)))
            context_dim = start_channels if i == 4 else start_channels * (2**(3-i))
            transformer_modules.append(
                TransformerLayer(
                dim = style_dim,
                pos_dim = style_dim,
                from_len = n_styles,
                to_len = context_res * context_res,
                from_dim = style_dim,
                to_dim = context_dim
            ))
            grid_pos, _ = get_positional_encoding(context_res, pos_dim=style_dim)
            self.register_buffer("grid_pos"+str(i), grid_pos)
            
        self.transformers  = nn.ModuleList(transformer_modules)
    
    def forward(self, x, contexts, eval=False, block=None):
        """
        Run concatenated styles through all attention layers, which computes attention 
        by comparing each style to each 'pixel' of context feature map.
        """
        shape = x.shape
        att_maps = []
        for i in range(5):
            c = contexts[i]
            c = c.reshape(c.shape[0], c.shape[1], -1).permute(0, 2, 1)
            x, att_map, _ = self.transformers[i](
                from_tensor = x, to_tensor = c, 
                from_pos = None,   to_pos = getattr(self, "grid_pos"+str(i)),
                dp=not eval, modify=True if block is None else (i != block)
            )
            att_maps.append(att_map)
        return x, att_maps