import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
import importlib
import models 
importlib.reload(models)  # Reload models to ensure they are up-to-date
from models import GeneratorRRDB, Discriminator, FeatureExtractor  # make sure these are modular
from math import log10
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

from types import SimpleNamespace
import math



class ESRGANLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):

            print("[DEBUG] Received config dict with keys:", config.keys())  # 
            config = SimpleNamespace(**config)
        self.save_hyperparameters()

        self.opt = config
        # self.generator = GeneratorRRDB(config.channels, filters=64, num_res_blocks=config.residual_blocks, num_upsample = int(math.log2(config.scale_factor)))
        self.generator = GeneratorRRDB(config.channels, filters=64, num_res_blocks=config.residual_blocks, num_upsample = config.scale_factor, use_ca=config.use_channel_attention)
        self.discriminator = Discriminator(input_shape=(config.channels, config.hr_height, config.hr_width), filters=config.discriminator_filters)
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.vgg19_54.load_state_dict(self._load_vgg_features(config.vgg19_path))
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_content = nn.L1Loss()
        self.criterion_pixel = nn.L1Loss()
        self.criterion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.automatic_optimization = False

    # def _load_vgg_features(self, path):
    #     fstate_dict = torch.load(path)
    # # Only keep features.* keys, strip the 'features.' prefix
    #     return {k.replace("features.", ""): v for k, v in fstate_dict.items() if k.startswith("features.")}
    def _load_vgg_features(self, path):
        state_dict = torch.load(path)

        # Case: saved with numeric keys like "0.weight", "1.bias"
        if all(k.split('.')[0].isdigit() for k in state_dict.keys()):
            return state_dict

        # Case: saved as full model with "features.X.weight"
        elif any(k.startswith("features.") for k in state_dict):
            return {
                k.replace("features.", ""): v
                for k, v in state_dict.items()
                if k.startswith("features.")
            }

        raise ValueError("Unexpected state_dict format")
    
    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers for the training.
        """
        # Create optimizers for the generator and discriminator
        opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.opt.lr, 
            betas=(self.opt.b1, self.opt.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.opt.lr, 
            betas=(self.opt.b1, self.opt.b2)
        )

        # Create learning rate schedulers for both optimizers
        scheduler_g = torch.optim.lr_scheduler.StepLR(
            opt_g, 
            step_size=self.opt.lr_scheduler_step_size, 
            gamma=self.opt.lr_scheduler_gamma
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            opt_d, 
            step_size=self.opt.lr_scheduler_step_size, 
            gamma=self.opt.lr_scheduler_gamma
        )

        # PyTorch Lightning expects a specific format when using schedulers.
        # return the optimizers 
        return (
            {'optimizer': opt_g, 'lr_scheduler': {'scheduler': scheduler_g, 'interval': 'step'}},
            {'optimizer': opt_d, 'lr_scheduler': {'scheduler': scheduler_d, 'interval': 'step'}}
        )

    def training_step(self, batch, batch_idx):
        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]

        opt_g, opt_d = self.optimizers()

        # === Forward generator and pixel loss ===
        gen_hr = self(imgs_lr)
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # === Warm-up phase (only pixel loss) ===
        if self.global_step < self.opt.warmup_batches:
            self.manual_backward(loss_pixel)
            opt_g.step()
            opt_g.zero_grad()
            self.log("train/loss_pixel", loss_pixel, prog_bar=True)
            return

        # === Generator update ===
        pred_real = self.discriminator(imgs_hr).detach()
        pred_fake = self.discriminator(gen_hr)

        valid = torch.ones_like(pred_fake)
        fake = torch.zeros_like(pred_fake)

        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        if self.opt.vgg19_channels == 3:
            gen_features = self.feature_extractor(gen_hr[:, :3, :, :])
            real_features = self.feature_extractor(imgs_hr[:, :3, :, :]).detach()
        else:
            gen_features = self.feature_extractor(gen_hr)
            real_features = self.feature_extractor(imgs_hr).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        loss_G = (
            self.opt.lambda_content * loss_content +
            self.opt.lambda_adv * loss_GAN +
            self.opt.lambda_pixel * loss_pixel
        )

        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()

        # === Discriminator update ===
        pred_real = self.discriminator(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
        loss_D = (loss_real + loss_fake) / 2

        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()

        # === Logging ===
        self.log_dict({
            "train/loss_G": loss_G,
            "train/loss_content": loss_content,
            "train/loss_pixel": loss_pixel,
            "train/loss_GAN": loss_GAN,
            "train/loss_D": loss_D,
        }, prog_bar=True)



    def validation_step(self, batch, batch_idx):
        imgs_lr = batch["lr"]
        imgs_hr = batch["hr"]
        gen_hr = self(imgs_lr)

        psnr = self.calculate_psnr(gen_hr, imgs_hr)
        ssim_val = self.criterion_ssim(gen_hr, imgs_hr)
        self.log("val/psnr", psnr, prog_bar=True)
        self.log("val/ssim", ssim_val, prog_bar=True)

        if batch_idx == 0:
            scale_factor = self.opt.scale_factor

            imgs_lr_up = F.interpolate(imgs_lr, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            img_grid = self.denormalize(torch.cat((imgs_lr_up, gen_hr), -1))
            
            images_saved_dir = self.opt.images_saved_dir
            os.makedirs(images_saved_dir, exist_ok=True)
            save_image(img_grid, f"{images_saved_dir}/step_{self.global_step}.png", nrow=1, normalize=False)

    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * log10(1 / mse.item()) if mse != 0 else float("inf")


    def denormalize(self, tensors):
        for c in range(3):
            tensors[:, c].mul_(self.opt.NORMALIZE_STD_HR[c]).add_(self.opt.NORMALIZE_MEAN_HR[c])
        return torch.clamp(tensors, 0, 255)


    def on_train_epoch_end(self):

        save_dir = self.opt.saved_models_dir
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.generator.state_dict(), os.path.join(save_dir, f"generator_epoch_{self.current_epoch}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{self.current_epoch}.pth"))
