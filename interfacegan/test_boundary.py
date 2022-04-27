import argparse
import os

from tqdm import tqdm
import numpy as np
import imageio

from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate
from models.model_settings import MODEL_POOL
from utils.logger import setup_logger


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


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')

    model = StyleGANGenerator(args.model_name, logger)

    latent_fname = "data/stylegan_celebahq/z.npy"
    boundary_fname = "boundaries/stylegan_celebahq_age/boundary.npy"

    latent_code = np.load(latent_fname)
    boundary_code = np.load(boundary_fname)
    
    for idx, z in enumerate(tqdm(latent_code)):
        interpolated_codes = linear_interpolate(z[None, :], boundary_code)
        interp_faces = []

        for interpolated_code in interpolated_codes:
            output_dict = model.easy_synthesize(interpolated_code[None, :])
            interp_faces.append(output_dict["image"].squeeze())

        output_img = np.concatenate(interp_faces, axis=1)
        imageio.imwrite(os.path.join(args.output_dir, f"{idx}.jpg"), output_img)

