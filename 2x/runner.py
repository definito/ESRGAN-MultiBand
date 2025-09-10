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


import data_generator

import models
import lightning_module

# Useful to reload modified external files without need
# of restarting the kernel. Just run again this cell.


reload(data_generator)


reload(models)
reload(lightning_module)


from data_generator import AllDataset,NormalizePair

from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from matplotlib import pyplot as plt
from math import log10
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning import Trainer
from lightning_module import ESRGANLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsummary import summary

gc.collect()
torch.cuda.empty_cache() 
torch.set_float32_matmul_precision('high') # Set precision for matmul operations to 'high' for better performance


# Set the seed for reproducibility
SEED = 42
BATCH_SIZE = 1
NUM_WORKERS = 4
ROOT_DIR = "/home/debwa/Desktop/courses/ss25/FINAL_DL/Dataset/Dataset_rgbi"


NORMALIZE_MEAN_LR = [0.0]*4
NORMALIZE_STD_LR  = [1.0]*4

NORMALIZE_MEAN_HR = [0.0]*4
NORMALIZE_STD_HR  = [1.0]*4




lr_transform = transforms.Compose([
    # transforms.Resize(FIXED_LR_SIZE),
    transforms.ToTensor()
])

hr_transform = transforms.Compose([
    # transforms.Resize(FIXED_HR_SIZE),
    transforms.ToTensor()])

train_dataset = AllDataset(
    root_dir=ROOT_DIR,
    split='train',
    transform=NormalizePair(NORMALIZE_MEAN_LR, NORMALIZE_STD_LR, NORMALIZE_MEAN_HR, NORMALIZE_STD_HR),
    augmentations=None
)
val_dataset = AllDataset(
    root_dir=ROOT_DIR,
    split='val',
    transform=NormalizePair(NORMALIZE_MEAN_LR, NORMALIZE_STD_LR, NORMALIZE_MEAN_HR, NORMALIZE_STD_HR),
    augmentations=None
)
test_dataset = AllDataset(
    root_dir=ROOT_DIR,
    split='test',
    transform=NormalizePair(NORMALIZE_MEAN_LR, NORMALIZE_STD_LR, NORMALIZE_MEAN_HR, NORMALIZE_STD_HR),
    augmentations=None
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)


config_dict = {
    'scale_factor': 2, # Scale factor for super-resolution
    'lr': 0.0002,
    'b1': 0.9,
    'b2': 0.999,
    'hr_height': 256,
    'hr_width': 256,
    'channels': 4,  # Number of channels in the input images
    'residual_blocks': 23, # Number of residual blocks in the generator
    'discriminator_filters': [64, 128, 128, 256, 256], # Filters for the discriminator
    'warmup_batches': 500,
    'lambda_adv': 0.001,
    'lambda_pixel': 0.02,
    'lambda_content': 0.4, # Content loss weight L1 weight
    'vgg19_path': "/home/debwa/Desktop/courses/ss25/FINAL_DL/Pretained/vgg19.pth",
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
    max_epochs=1,
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
