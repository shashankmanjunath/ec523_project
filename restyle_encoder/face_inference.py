from argparse import Namespace
import pprint
import time
import os

from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import torch
import dlib

from scripts.align_faces_parallel import align_face
from utils.inference_utils import run_on_batch
from utils.common import tensor2im
from models.psp import pSp


def run_alignment(image_path):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0), input_code=True, randomize_noise=False, return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_coupled_results(result_batch, transformed_image, opts):
    """
    Visualize output images from left to right (the input image is on the right)
    """
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    result_tensors = result_batch[0]  # there's one image in our batch
    result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
    input_im = tensor2im(transformed_image)
    res = np.array(result_images[0].resize(resize_amount))

    for idx, result in enumerate(result_images[1:]):
        res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)

    res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
    res = Image.fromarray(res)
    return res


if __name__ == "__main__":
    model_path = "../pretrained_models/psp_stylegan2_attention_ffhq_48k.pt"

    image_path = "notebooks/images/face_img.jpg"
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    pprint.pprint(opts)

    # update the training options
    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval().cuda()
    print('Model successfully loaded!')

    print("Aligning faces...")
    input_image = run_alignment(image_path)
    transformed_image = img_transforms(input_image)

    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolutio

    with torch.no_grad():
        avg_image = get_avg_image(net)
        tic = time.time()
        result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    res = get_coupled_results(result_batch, transformed_image, opts)
    res.save(f"./runs/results.jpg")
