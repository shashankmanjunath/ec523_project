"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import torch

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from models.psp import pSp
from models.psp_gansformer import pSpGansformer
from configs import data_configs, transforms_config
from datasets.images_dataset import ImagesDataset
from models.gansformer.training import misc


def main():
	opts = TrainOptions().parse()
	os.makedirs(opts.exp_dir, exist_ok=True)
	os.makedirs(opts.exp_dir+"_out", exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	device = opts.device

	net = pSpGansformer(opts) if opts.use_gansformer else pSp(opts)
	net = net.to(device)

	data = configure_data(opts)

	for i,d in enumerate(data):
		img = d[0].unsqueeze(0).to(device)
		out = net.forward(img).cpu().detach().numpy()
		misc.to_pil(out[0]).save("{}_out/{}.png".format(opts.exp_dir, i))
		for j in range(5):
			out = net.forward(img, block_attention=j).cpu().detach().numpy()
			misc.to_pil(out[0]).save("{}_out/{}-{}.png".format(opts.exp_dir, i, j))

	print("done")


def configure_data(opts):
	if opts.dataset_type not in data_configs.DATASETS.keys():
		Exception(f'{self.opts.dataset_type} is not a valid dataset_type')

	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = transforms_config.EncodeTransforms(opts).get_transforms()
	data = ImagesDataset(source_root=opts.exp_dir,
									target_root=opts.exp_dir,
									source_transform=transforms_dict['transform_source'],
									target_transform=transforms_dict['transform_test'],
									opts=opts)
	return data

if __name__ == '__main__':
	main()

#    python scripts/attention_blocking.py --dataset_type=ffhq_encode --encoder_type=GradualStyleEncoder --use_attention --exp_dir=experiment/att_maps --start_from_latent_avg --input_nc=3 --output_size=256 --checkpoint_path=experiment/psp_stylegan_attention_ffhq_encode7/checkpoints/best_model.pt