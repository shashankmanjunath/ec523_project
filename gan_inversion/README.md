# GAN Inversion with Attention

Code modified from ReStyle, available at https://github.com/yuval-alaluf/restyle-encoder. See requirements.txt in parent directory for all necessary packages.

## Pretrained Models
Our two final models we trained for the GAN Inversion task, along with the base pSp we compare against are available below:
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN2-based Attention](https://drive.google.com/file/d/1JjJtg4ehHOsn86OrFx9Xdj0Fa1jMqyL2/view?usp=sharing) | pSp + StyleGAN2 model with added attention layers, trained on FFHQ with 256x256 output resolution.
|[FFHQ GANformer-based Attention](https://drive.google.com/file/d/1JjJtg4ehHOsn86OrFx9Xdj0Fa1jMqyL2/view?usp=sharing) | pSp + GANformer model with added attention layers, trained on FFHQ with 256x256 output resolution.
|[Base pSp](https://drive.google.com/file/d/1JjJtg4ehHOsn86OrFx9Xdj0Fa1jMqyL2/view?usp=sharing) | pSp model from the original paper, trained on FFHQ with 256x256 output resolution.

### Auxiliary Models
Various auxiliary models are needed to train our models, which we download and put under "pretrained_models/". 
This includes the StyleGAN and GANformer generators and several models used for loss computation.

| Path | Description
| :--- | :----------
|[FFHQ StyleGAN2](https://drive.google.com/file/d/1nxlaQtJ536D7pk1chk05q5M_oVrIYp9g/view?usp=sharing) | StyleGAN2 model trained on FFHQ with 256x256 output resolution, from the [Rosinality PyTorch implementation](https://github.com/rosinality/stylegan2-pytorch).
|[FFHQ GANformer](https://drive.google.com/file/d/1nxlaQtJ536D7pk1chk05q5M_oVrIYp9g/view?usp=sharing) | GANformer model trained on FFHQ with 256x256 output resolution, provided by [GANformer](https://github.com/dorarad/gansformer) [here](https://drive.google.com/uc?id=1tgs-hHaziWrh0piuX3sEd8PwE9gFwlNh).
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss and encoder backbone on human facial domain.
|[ResNet-34 Model](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | ResNet-34 model trained on ImageNet taken from [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) for initializing our encoder backbone.
|[MoCov2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view) | Pretrained ResNet-50 model trained using MOCOv2 for computing MoCo-based loss on non-facial domains. The model is taken from the [official implementation](https://github.com/facebookresearch/moco).
|[CurricularFace Backbone](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view?usp=sharing) | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation.
|[MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view?usp=sharing) | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)

## Training
### Datasets
We provide the FFHQ and CelebAHQ datasets used for training and testing, which were reduced to 256x256 for time efficiency.
| Path | Description
| :--- | :----------
|[FFHQ](https://drive.google.com/file/d/1kbYvkO85ZcgXRbMSArrK5ZtC_RpD45n7/view?usp=sharing) | FFHQ in 256x256 resolution, modified from [NVlabs](https://github.com/NVlabs/ffhq-dataset).
|[CelebAHQ](https://drive.google.com/file/d/1kbYvkO85ZcgXRbMSArrK5ZtC_RpD45n7/view?usp=sharing) | CelebAHQ in 256x256 resolution, modified from [CelebAMaskHQ](https://github.com/switchablenorms/CelebAMask-HQ).

### Preparing your Data
Update `configs/paths_config.py` with paths to the FFHQ and CelebA datasets, used for training and testing. Default settings are:
```
dataset_paths = {
	'ffhq': os.path.expanduser('~/ffhq-256/'),
	'celeba_test': os.path.expanduser('~/CelebA-HQ/')
}
```
### Training Scripts
The main training script is `scripts/train_psp.py` Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs. To train each model with the default settings, use the following commands (with optional checkpoint path for continuing training):

- StyleGAN2-based with Attention:
```
python scripts/train_psp.py \
--dataset_type=ffhq_encode \
--exp_dir=experiment/psp_stylegan_attention_ffhq_encode \
--use_attention \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--id_lambda=0.1 \
--input_nc=3 \
--output_size=256 \
--stylegan_weights=pretrained_models/stylegan2-ffhq-config-f.pt \
--checkpoint_path=best_model.pt 
```

- GANformer-based with Attention:
```
python scripts/train_psp.py \
--dataset_type=ffhq_encode \
--exp_dir=experiment/psp_attention_ffhq_encode \
--use_gansformer \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--id_lambda=0.1 \
--input_nc=3 \
--output_size=256 \
--stylegan_weights=pretrained_models/ffhq-gansformer-256.pkl \
--checkpoint_path=best_model.pt 
```

- Original pSp:
```
python scripts/train_psp.py \
--dataset_type=ffhq_encode \
--exp_dir=experiment/psp_stylegan_attention_ffhq_encode \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--id_lambda=0.1 \
--input_nc=3 \
--stylegan_weights=pretrained_models/stylegan2-ffhq-config-f.pt \
--checkpoint_path=best_model.pt 
```

## Testing
### Loss Evaluation
Run python scripts/loss_evaluation.py with the same arguments that would be used for training and checkpoint_path as the model to evaluate.

## Repository structure
| Path | Description <img width=200>
| :--- | :---
| gan_inversion | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in original ReStyle
| &boxvr;&nbsp; licenses | Folder containing licenses of the open source projects used in this repository
| &boxvr; models | Folder containing all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing various architecture implementations and attention implementation modified from [GANformer](https://github.com/dorarad/gansformer)
| &boxv;&nbsp; &boxvr;&nbsp; mtcnn | MTCNN implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; stylegan | StyleGAN model from [huangzh13](https://github.com/huangzh13/StyleGAN.pytorch), used for comparison
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; psp.py | Implementation of pSp encoder with optional attention
| &boxv;&nbsp; &boxvr;&nbsp; psp_gansformer.py | Implementation of pSp encoder for the GANformer decoder
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; scripts | Folder with running scripts for training, inference, and metric computations
| &boxvr;&nbsp; training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxvr;&nbsp; utils | Folder with various utility functions
| <img width=300> | <img>


## Credits
**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

**StyleGAN2 model and implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**GANformer model and implementation:**  
https://github.com/dorarad/gansformer  
Copyright (c) 2021 Drew Arad Hudson
License (MIT) https://github.com/dorarad/gansformer/blob/main/LICENSE  

**IR-SE50 model and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS model and implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE 

**ReStyle model and implementation:**   
https://github.com/yuval-alaluf/restyle-encoder
Copyright (c) 2021 Yuval Alaluf  
License (MIT) https://github.com/yuval-alaluf/restyle-encoder/blob/main/LICENSE

**Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/eladrich/pixel2style2pixel/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and 
[ReStyle](https://github.com/yuval-alaluf/restyle-encoder).
