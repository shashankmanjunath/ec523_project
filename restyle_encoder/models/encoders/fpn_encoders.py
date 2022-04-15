import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from models.encoders.helpers import get_block, get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock


class GradualStyleAttentionEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None, dim=544, attention=True):
        super(GradualStyleAttentionEncoder, self).__init__()
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

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(dim, dim, 16, attention=attention, context_res=16, context_dim=512)
            elif i < 5:
                style = GradualStyleBlock(dim, dim, 32, attention=attention, context_res=16, context_dim=512)
            elif i < self.middle_ind:
                style = GradualStyleBlock(dim, dim, 32, attention=attention, context_res=32, context_dim=256)
            elif i < 9:
                style = GradualStyleBlock(dim, dim, 64, attention=attention, context_res=64, context_dim=128)
            elif i < 11:
                style = GradualStyleBlock(dim, dim, 64, attention=attention, context_res=128, context_dim=64)
            else:
                style = GradualStyleBlock(dim, dim, 64, attention=attention, context_res=256, context_dim=64)
            self.styles.append(style)
        self.latlayer0 = nn.Conv2d(512, dim, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(256, dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, dim, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
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

        # bu3: 16 x 16 x 512 -> 16 x 16 x 544 (map)
        td3 = self.latlayer0(bu3)
        for j in range(self.coarse_ind): # 1-3
            latents.append(self.styles[j](td3, res4))

        # bu2: 32 x 32 x 256 -> 32 x 32 x 544 (map)
        # td3: 16 x 16 x 544 -> 32 x 32 x 544 (upsample)
        td2 = self._upsample_add(td3, self.latlayer1(bu2))
        for j in range(self.coarse_ind, self.middle_ind):
            if j < 5: # 4-5
                latents.append(self.styles[j](td2, res4))
            else: # 6-7
                latents.append(self.styles[j](td2, res3))

        # bu1: 64 x 64 x 128 -> 64 x 64 x 544 (map)
        # td2: 32 x 32 x 544 -> 64 x 64 x 544 (upsample)
        td1 = self._upsample_add(td2, self.latlayer2(bu1))
        for j in range(self.middle_ind, self.style_count):
            if j < 9: # 8-9
                latents.append(self.styles[j](td1, res2))
            elif j < 11: # 10-11
                latents.append(self.styles[j](td1, res1))
            else: # 12-15
                latents.append(self.styles[j](td1, res0))
        out = torch.stack(latents, dim=1)
        return out

class GradualStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(GradualStyleEncoder, self).__init__()
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
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
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
