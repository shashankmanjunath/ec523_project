import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from models.encoders.helpers import get_block, get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock, GansformerStyleBlock, AttentionBlock


class GradualStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None, input_res=256):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.attention = None
        if opts.use_attention:
            self.attention = AttentionBlock(input_res, 512, n_styles)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, return_att=False, block=None):
        # bu -> bottom up pathway, td -> top down pathway
        # h -> from tensor in attention, c -> to tensor (context)
        x = self.input_layer(x) # (256, 256)
        res4 = x 
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2: # (128, 128)
                res3 = x
            elif i == 6: # (64, 64)
                res2 = bu1 = x
            elif i == 20: # (32, 32)
                res1 = bu2 = x
            elif i == 23: # (16, 16)
                res0 = bu3 = x

        # bu3: 16 x 16 x 512
        td3 = bu3
        for j in range(self.coarse_ind): # 1-3
            latents.append(self.styles[j](td3))

        # bu2: 32 x 32 x 256 -> 32 x 32 x 512 (map)
        # td3: 16 x 16 x 512 -> 32 x 32 x 512 (upsample)
        td2 = self._upsample_add(td3, self.latlayer1(bu2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](td2))

        # bu1: 64 x 64 x 128 -> 64 x 64 x 512 (map)
        # td2: 32 x 32 x 512 -> 64 x 64 x 512 (upsample)
        td1 = self._upsample_add(td2, self.latlayer2(bu1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](td1))
        out = torch.stack(latents, dim=1)

        if self.attention is not None:
            context_features = [res0, res1, res2, res3, res4]
            out, att_maps = self.attention(out, context_features, eval=return_att, block=block)

        if return_att:
            return out, att_maps
        return out

class GansformerStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(GansformerStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        style_dim = 544
        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GansformerStyleBlock(style_dim, style_dim, 16, context_res=16, context_dim=512)
            elif i < 5:
                style = GansformerStyleBlock(style_dim, style_dim, 32, context_res=16, context_dim=512)
            elif i < self.middle_ind:
                style = GansformerStyleBlock(style_dim, style_dim, 32, context_res=32, context_dim=256)
            elif i < 9:
                style = GansformerStyleBlock(style_dim, style_dim, 64, context_res=64, context_dim=128)
            elif i < 11:
                style = GansformerStyleBlock(style_dim, style_dim, 64, context_res=128, context_dim=64)
            else:
                style = GansformerStyleBlock(style_dim, style_dim, 64, context_res=256, context_dim=64)
            self.styles.append(style)
        self.latlayer0 = nn.Conv2d(512, style_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(256, style_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, style_dim, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, return_att=False):
        # bu -> bottom up pathway, td -> top down pathway
        # h -> from tensor in attention, c -> to tensor (context)
        x = self.input_layer(x) # (256, 256)
        res0 = x 
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2: # (128, 128)
                res1 = x
            elif i == 6: # (64, 64)
                res2 = bu1 = x
            elif i == 20: # (32, 32)
                res3 = bu2 = x
            elif i == 23: # (16, 16)
                res4 = bu3 = x

        att_maps = []
        # bu3: 16 x 16 x 512 -> 16 x 16 x 544 (map)
        td3 = self.latlayer0(bu3)
        for j in range(self.coarse_ind): # 1-3
            style, att_map = self.styles[j](td3, res4)
            latents.append(style)
            att_maps.append(att_map)

        # bu2: 32 x 32 x 256 -> 32 x 32 x 544 (map)
        # td3: 16 x 16 x 544 -> 32 x 32 x 544 (upsample)
        td2 = self._upsample_add(td3, self.latlayer1(bu2))
        for j in range(self.coarse_ind, self.middle_ind):
            if j < 5: # 4-5
                style, att_map = self.styles[j](td2, res4)
            else: # 6-7
                style, att_map = self.styles[j](td2, res3)
            latents.append(style)
            att_maps.append(att_map)

        # bu1: 64 x 64 x 128 -> 64 x 64 x 544 (map)
        # td2: 32 x 32 x 544 -> 64 x 64 x 544 (upsample)
        td1 = self._upsample_add(td2, self.latlayer2(bu1))
        for j in range(self.middle_ind, self.style_count):
            if j < 9: # 8-9
                style, att_map = self.styles[j](td1, res2)
            elif j < 11: # 10-11
                style, att_map = self.styles[j](td1, res1)
            else: # 12-15
                style, att_map = self.styles[j](td1, res0)
            latents.append(style)
            att_maps.append(att_map)
        out = torch.stack(latents, dim=1)
        if return_att:
            return out, att_maps
        return out

class ResNetGradualStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet34 backbone.
    """
    def __init__(self, n_styles=18, opts=None):
        super(ResNetGradualStyleEncoder, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)

        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 12:
                c2 = x
            elif i == 15:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out
