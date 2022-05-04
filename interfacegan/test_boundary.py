import argparse
import os

from tqdm import tqdm
import numpy as np
import imageio
import torch

from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate
from models.model_settings import MODEL_POOL
from utils.logger import setup_logger
from models.psp import pSp


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Generate images with given model.')
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=list(MODEL_POOL),
            help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
            help='Directory to save the output results.  (required)')
    parser.add_argument('-i', '--latent_codes_path', type=str, default='',
            help='If specified, will load latent codes from given ' 'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=1,
            help='Number of images to generate. This field will be ignored if `latent_codes_path` is specified. ' '(default: 1)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='z',
            choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'], help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-S', '--generate_style', action='store_true',
            help='If specified, will generate layer-wise style codes ' 'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
            help='If specified, will skip generating images in ' 'Style GAN. (default: generate images)')
    return parser.parse_args()


def convert_to_opts(args):
    model_path = MODEL_POOL[args.model_name]["model_path"]
    opts = torch.load(model_path, map_location=torch.device("cuda:0"))["opts"]
    opts['checkpoint_path'] = model_path
    opts['device'] = "cuda:0"
    opts = argparse.Namespace(**opts)
    return opts

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')
    opts = convert_to_opts(args)

    #  model = StyleGAN2Generator(args.model_name, logger)
    model = pSp(opts)

    latent_fname = "data/psp_stylegan2_ffhq/z.npy"
    boundary_fname = "boundaries/psp_stylegan2_ffhq_age/boundary.npy"

    latent_code = np.load(latent_fname)
    boundary_code = np.load(boundary_fname)
    
    for idx, z in enumerate(tqdm(latent_code)):
        interpolated_codes = linear_interpolate(z[None, :], boundary_code)
        interp_faces = []

        for interpolated_code in interpolated_codes:
            imgs = model(torch.as_tensor(interpolated_code[None, :]).cuda(), input_code=True)
            interp_faces.append(imgs.squeeze().detach().cpu().numpy())

        interp_faces = [np.transpose(x, axes=[1, 2, 0]) for x in interp_faces]
        output_img = np.concatenate(interp_faces, axis=1)
        output_img = convert(output_img, 0, 255, np.uint8)
        imageio.imwrite(os.path.join(args.output_dir, f"{idx}.jpg"), output_img.astype(np.uint8))

