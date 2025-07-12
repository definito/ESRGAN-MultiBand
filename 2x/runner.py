import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from importlib import reload
import datasets

import data_generator
import utils
import models
import lightning_module
# import trainer
# Useful to reload modified external files without need
# of restarting the kernel. Just run again this cell.
reload(datasets)

reload(data_generator)
reload(utils)

reload(models)
reload(lightning_module)

from datasets import *
from data_generator import SRDataset,NormalizePair,ComposePair

from utils import prepare_dataloaders
from models import *
# from datasets import *


import torch.nn as nn
import torch.nn.functional as F
import torch

import gc
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log10
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning import Trainer
from lightning_module import ESRGANLightning
from pytorch_lightning.callbacks import ModelCheckpoint


gc.collect()
torch.cuda.empty_cache() 
torch.set_float32_matmul_precision('high') # Set precision for matmul operations to 'high' for better performance


# Set the seed for reproducibility
SEED = 42
BATCH_SIZE = 1
NUM_WORKERS = 4
MANIFEST_PATH = "/home/debwa/Desktop/courses/ss25/dl/final_project_dataset_generator/Dataset-2348-2x-10m-05m/data_map.csv"
ROOT_DIR = "/home/debwa/Desktop/courses/ss25/dl/final_project_dataset_generator/Dataset-2348-2x-10m-05m"

NORMALIZE_MEAN_LR = [0.0, 0.0, 0.0, 0.0]
NORMALIZE_STD_LR  = [1.0, 1.0, 1.0, 1.0]

NORMALIZE_MEAN_HR = [0.0, 0.0, 0.0, 0.0]
NORMALIZE_STD_HR  = [1.0, 1.0, 1.0, 1.0]

# _,_,_, stats = prepare_dataloaders(
#     manifest_path=MANIFEST_PATH,
#     root_dir=ROOT_DIR,
#     seed=SEED,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     normalize_means_stds=None,  # or pass precomputed if available
#     load_and_split_manifest_fn=load_and_split_manifest,
#     SRDataset_class=SRDataset,
#     ComposePair_class=ComposePair,
#     NormalizePair_class=NormalizePair
# )



train_loader, val_loader, test_loader, stats = prepare_dataloaders(
    manifest_path=MANIFEST_PATH,
    root_dir=ROOT_DIR,
    seed=SEED,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    normalize_means_stds=(NORMALIZE_MEAN_LR, NORMALIZE_STD_LR, NORMALIZE_MEAN_HR, NORMALIZE_STD_HR),  # or pass precomputed if available
    load_and_split_manifest_fn=load_and_split_manifest,
    SRDataset_class=SRDataset,
    ComposePair_class=ComposePair,
    NormalizePair_class=NormalizePair
)


def visualize_rgb_pair(lr_tensor, hr_tensor, title_lr="Low-Resolution", title_hr="High-Resolution"):
    def stretch(img):
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        return np.clip(img.transpose(1, 2, 0), 0, 1)

    lr_img = stretch(lr_tensor.cpu().numpy())
    hr_img = stretch(hr_tensor.cpu().numpy())

    plt.figure(figsize=(12, 5))
    for i, (img, title) in enumerate(zip([lr_img, hr_img], [title_lr, title_hr])):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.title(f"{title}\n{img.shape[:2]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

batch = next(iter(train_loader))
visualize_rgb_pair(batch['lr'][0], batch['hr'][0])

config_dict = {
    'scale_factor': 2,
    'lr': 0.0002,
    'b1': 0.9,
    'b2': 0.999,
    'hr_height': 256,
    'hr_width': 256,
    'channels': 4,  # Number of channels in the input images
    'residual_blocks': 5,
    'discriminator_filters': [64, 128, 256, 512], # Filters for the discriminator
    'warmup_batches': 500,
    'lambda_adv': 0.005,
    'lambda_pixel': 0.01,
    'lambda_content': 0.01, # Content loss weight L1 weight
    'vgg19_path': "../Pretrained/vgg19.pth",
    'NORMALIZE_MEAN_HR': NORMALIZE_MEAN_HR,
    'NORMALIZE_STD_HR': NORMALIZE_STD_HR,
    'images_saved_dir': "images/validation",
    'saved_models_dir': "saved_models",
}




model = ESRGANLightning(config_dict)

checkpoint_callback = ModelCheckpoint(
    monitor='val/psnr',
    dirpath='saved_models',
    filename='model_{epoch:02d}_psnr_{val/psnr:.2f}',
    save_top_k=5,
    mode='max'
)
logger = CSVLogger("logs", name="esrgan_experiment")
trainer = Trainer(
    default_root_dir=os.getcwd(),
    max_epochs=200,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
    logger=logger,
    callbacks=[checkpoint_callback]
)
# CKPT_PATH = None  # Path to a checkpoint if you want to resume training
# trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=CKPT_PATH)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# test_model = ESRGANLightning(config_dict)
# generator_path = "saved_models/generator_epoch_2.pth" 
# test_model.generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
