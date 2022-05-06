# Generate images using pretrained network pickle.
import os
import numpy as np
import PIL.Image
from tqdm import trange 
import argparse

import models.gansformer.dnnlib as dnnlib
import torch
from models.stylegan.GAN import Generator as StyleGANGenerator

from models.gansformer.training import misc
from models.gansformer.training.misc import crop_max_rectangle as crop

# Generate images using pretrained network pickle.
def run(model, gpus, output_dir, images_num, truncation_psi, ratio):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus                             # Set GPUs
    device = torch.device("cuda")

    print("Loading networks...")
    G = StyleGANGenerator(1024, blur_filter=[1,2,1], truncation_psi=0).to(device)
    G.load_state_dict(torch.load(model)["state_dict"], strict=True)

    print("Generate and save images...")
    os.makedirs(output_dir, exist_ok = True)                              # Make output directory

    for i in trange(images_num):
        # z = torch.randn([1, *G.input_shape[1:]], device = device)         # Sample latent vector
        w_avg = G.mean_latent(int(1e5), device)[0].detach()
        print(w_avg.shape)
        dw = torch.randn([1, 18, 512], device = device) / 5
        imgs = G(w_avg+dw, depth=int(np.log2(1024)) - 2, alpha=1, input_is_latent=True)[0].detach().cpu().numpy()     # Generate an image
        pattern = "{}/sample_{{:06d}}.png".format(output_dir)             # Output images pattern
        img = crop(misc.to_pil(imgs[0]), ratio).save(pattern.format(i))   # Save the image

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume", type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    # Pretrained models' ratios: CLEVR (0.75), Bedrooms (188/256), Cityscapes (0.5), FFHQ (1.0)
    args, _ = parser.parse_known_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
