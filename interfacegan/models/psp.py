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

        self.batch_size = 1
        self.run_device = torch.device("cuda:0")

        # Define architecture
        self.encoder = self.set_encoder()
        if self.opts.use_stylegan:
            self.decoder = StyleGANGenerator(self.opts.output_size, blur_filter=[1,2,1], truncation_psi=0)
        else:
            self.decoder = StyleGAN2Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

        self.encoder = self.encoder.to(self.run_device)
        self.decoder = self.decoder.to(self.run_device)

    def set_encoder(self):
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
            print(f'Loading ReStyle pSp from checkpoint: {self.opts.checkpoint_path}')
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
                inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
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

    def postprocess(self, images):
        """Postprocesses the output images if needed.

        This function assumes the input numpy array is with shape [batch_size,
        channel, height, width]. Here, `channel = 3` for color image and
        `channel = 1` for grayscale image. The return images are with shape
        [batch_size, height, width, channel]. NOTE: The channel order of output
        image will always be `RGB`.

        Args:
          images: The raw output from the generator.

        Returns:
          The postprocessed images with dtype `numpy.uint8` with range [0, 255].

        Raises:
          ValueError: If the input `images` are not with type `numpy.ndarray` or not
            with shape [batch_size, channel, height, width].
        """
        if not isinstance(images, np.ndarray):
            raise ValueError(f'Images should be with type `numpy.ndarray`!')

        images_shape = images.shape
        if len(images_shape) != 4 or images_shape[1] not in [1, 3]:
            raise ValueError(f'Input should be with shape [batch_size, channel, '
                             f'height, width], where channel equals to 1 or 3. '
                             f'But {images_shape} is received!')
        max_val = 1.0
        min_val = -1.0 
        images = (images - min_val) * 255 / (max_val - min_val)
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        #  if self.channel_order == 'BGR':
        #      images = images[:, :, :, ::-1]

        return images

    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = np.random.randn(num, 512)
        elif latent_space_type == 'W':
            latent_codes = np.random.randn(num, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')
        return latent_codes.astype(np.float32)

    def preprocess(self, latent_codes, latent_space_type='Z'):
        """Preprocesses the input latent code if needed.

        Args:
          latent_codes: The input latent codes for preprocessing.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          The preprocessed latent codes which can be used as final input for the
            generator.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        latent_space_type = latent_space_type.upper()
        #  if latent_space_type == 'Z':
        #      latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
        #      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
        #      latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
        #  elif latent_space_type == 'W':
        #      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
        #  elif latent_space_type == 'WP':
        #      latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
        #  else:
        #      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)

    def easy_sample(self, num, latent_space_type='Z'):
        return self.preprocess(self.sample(num, latent_space_type),
                               latent_space_type)

    def synthesize(self, latent_codes, latent_space_type='Z', generate_style=False, generate_image=True):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        results = {}

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape
        # Generate from Z space.
        if latent_space_type == 'Z':
            w_avg = self.decoder.mean_latent(1, device=self.run_device)[0].detach().cpu().numpy()
            dw = latent_codes
            ws = w_avg + latent_codes
            #  wps = self.model.truncation(ws)
            results['z'] = latent_codes
            results['w'] = self.get_value(ws)
            #  results['wp'] = self.get_value(wps)
        # Generate from W space.
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            ws = ws.to(self.run_device)
            results['w'] = latent_codes
            #  results['wp'] = self.get_value(wps)
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            wps = wps.to(self.run_device)
            results['wp'] = latent_codes
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        #  if generate_style:
        #      for i in range(self.num_layers):
        #          style = self.model.synthesis.__getattr__(
        #              f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
        #          results[f'style{i:02d}'] = self.get_value(style)

        if generate_image:
            images = self.decoder(torch.as_tensor(ws).to(self.run_device))[0]
            #  images = self.forward(wps)
            results['image'] = self.get_value(images)

        return results

    def easy_synthesize(self, latent_codes, **kwargs):
        """Wraps functions `synthesize()` and `postprocess()` together."""
        outputs = self.synthesize(latent_codes, **kwargs)
        if 'image' in outputs:
            outputs['image'] = self.postprocess(outputs['image'])

        return outputs

    def get_batch_inputs(self, latent_codes):
        """Gets batch inputs from a collection of latent codes.

        This function will yield at most `self.batch_size` latent_codes at a time.

        Args:
          latent_codes: The input latent codes for generation. First dimension
            should be the total number.
        """
        total_num = latent_codes.shape[0]
        for i in range(0, total_num, self.batch_size):
            yield latent_codes[i:i + self.batch_size]

    def get_value(self, tensor):
        """Gets value of a `torch.Tensor`.

        Args:
          tensor: The input tensor to get value from.

        Returns:
          A `numpy.ndarray`.

        Raises:
          ValueError: If the tensor is with neither `torch.Tensor` type or
            `numpy.ndarray` type.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return tensor.to(torch.device("cpu")).detach().numpy()
        raise ValueError(f'Unsupported input type `{type(tensor)}`!')

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
