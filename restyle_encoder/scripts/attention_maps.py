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

	net = pSp(opts)
	net = net.to(device)

	data = configure_data(opts)

	pattern = "{}_out/{{}}-{{}}.png".format(opts.exp_dir) 
	for d in data:
		img = d[0].unsqueeze(0).to(device)
		att_maps = net.forward(img, return_att=True)
		for i,att_map in enumerate(att_maps):
			att_map = torch.nn.functional.softmax(att_map, dim=2)
			for j,style_att in enumerate(torch.split(att_map, 1, dim=2)):
				style_att = (style_att - torch.mean(style_att)) / torch.std(style_att)
				out = style_att.reshape(int(style_att.shape[3]**0.5), int(style_att.shape[3]**0.5)).cpu().detach().numpy()
				misc.to_pil(out).save(pattern.format(i,j))

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

#    python scripts/attention_maps.py --dataset_type=ffhq_encode --encoder_type=GradualStyleEncoder --use_attention --exp_dir=experiment/att_maps --start_from_latent_avg --input_nc=3 --output_size=256 --checkpoint_path=experiment/psp_stylegan_attention_ffhq_encode7/checkpoints/best_model.pt