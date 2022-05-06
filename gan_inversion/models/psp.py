"""
This file defines the core research contribution
"""
import math
import numpy as np
import torch
from torch import nn

from models.stylegan.GAN import Generator as StyleGANGenerator
from models.stylegan2.model import Generator as StyleGAN2Generator
from models.stylegan.loader import load as load_stylegan
from configs.paths_config import model_paths
from models.encoders import fpn_encoders, restyle_psp_encoders
from utils.model_utils import RESNET_MAPPING


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()

        # StyleGAN 1 and 2 supported for comparison
        if self.opts.use_stylegan:
            self.decoder = StyleGANGenerator(self.opts.output_size, blur_filter=[1,2,1], truncation_psi=0)
        else:
            self.decoder = StyleGAN2Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        # only GradualStyleEncoder supports attention
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = fpn_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetGradualStyleEncoder':
            encoder = fpn_encoders.ResNetGradualStyleEncoder(self.n_styles, self.opts)
        elif self.opts.encoder_type == 'BackboneEncoder':
            encoder = restyle_psp_encoders.BackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetBackboneEncoder':
            encoder = restyle_psp_encoders.ResNetBackboneEncoder(self.n_styles, self.opts)
        else:
            raise Exception(f'{self.opts.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading pSp from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            encoder_ckpt = self.__get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
            decoder_ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(decoder_ckpt['g_ema'], strict=False)
            self.__load_latent_avg(decoder_ckpt, repeat=self.n_styles)

    def forward(self, x, latent=None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_att=False, return_latents=False, alpha=None, 
                average_code=False, input_is_full=False, block_attention=None):

        # Used to view attention maps
        if return_att:
            if not self.opts.use_attention or self.opts.encoder_type != 'GradualStyleEncoder':
                raise Exception('Attention maps requires use_attention and GradualStyleEncoder')
            _, att_maps = self.encoder(x, return_att=True)
            return att_maps
        elif input_code:
            codes = x
        else:
            codes = self.encoder(x, block=block_attention)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if average_code:
            input_is_latent = True
        else:
            input_is_latent = (not input_code) or (input_is_full)

        # Call StyleGAN1 or StyleGAN2 decoder
        if self.opts.use_stylegan:
            images, result_latent = self.decoder(codes,
                                                depth=int(np.log2(self.opts.output_size)) - 2,
                                                alpha=1,
                                                input_is_latent=input_is_latent,
                                                return_latents=return_latents)
        else:
            images, result_latent = self.decoder([codes],
                                                input_is_latent=input_is_latent,
                                                randomize_noise=randomize_noise,
                                                return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_encoder_checkpoint(self):
        if "ffhq" in self.opts.dataset_type:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'], map_location='cpu')
            # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            return encoder_ckpt
        else:
            print('Loading encoders weights from resnet34!')
            encoder_ckpt = torch.load(model_paths['resnet34'], map_location='cpu')
            # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['conv1.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                encoder_ckpt['conv1.weight'] = altered_input_layer
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt