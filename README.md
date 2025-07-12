# ESRGAN-MultiBand (Repository is being updated)

**ESRGAN-MultiBand** is a PyTorch Lightning-based implementation of the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN), adapted for multispectral satellite imagery. Unlike typical RGB applications, this version supports 4+ spectral bands (e.g., near-infrared, red-edge) commonly used in remote sensing datasets.

>  This repository is currently being updated. Finalized documentation, pretrained models, and all training scripts will be added soon.


# SEN12-VENUS ESRGAN Super-Resolution (2x)
This repository implements a complete PyTorch Lightning pipeline for super-resolution on the SEN12-VENUS dataset, using an enhanced ESRGAN architecture with 4-band (RGB+IR) input.

**_Dataset Tested:_ DIV2k(RGB- small subset)_2x , 2x_sen2venus RGB only, 2x_sen2vusRGBI,  4x sen2venus**

| Dataset           | Input Bands | Epochs | PSNR (dB)      | SSIM    |
|------------------|--------|--------|----------------|---------|
| DIV2K-2x (subset) | RGB    | 200    | 30.4836        | 0.8693  |
| Sen2Venus-2x      | RGB    | 63     | 41.0543        | 0.9633  |
| Sen2Venus-2x      | RGBI   | 63     | 31.2548        | 0.9281  |
| Sen2Venus-4x      | 5678   | 76     | 36.40      | 0.928 |


# Key Features:
2× Super-Resolution using a GAN-based architecture with RRDB generator and VGG19-based perceptual loss.

4-channel support: RGB + Infrared input/output.

Built-in dataloader with rasterio support for GeoTIFFs.

Pixel, Content, GAN, and SSIM Losses for balanced training.

Lightning-ready training loop with checkpointing, logging, and visual validation image saving.

Easy-to-use runner with custom normalization, reproducible split, and visualization.

# Coming Soon:
4× Super-Resolution variant with multi-level upscaling.

Band 5,6,7,8 ESRGAN model for multispectral learning beyond RGB-I.

### **DIV2K:** 

![Image](https://github.com/user-attachments/assets/7658cdd1-80d4-47f9-9563-b04cc914382d)

![Image](https://github.com/user-attachments/assets/4cd3072b-8ca4-401c-afd5-e6125d73ec68)

![Image](https://github.com/user-attachments/assets/ad1db13f-3a86-40ab-9a72-0249ed0a1464)


# Result for 2x 
### **SEN2VENUS_2x - band- BGIR** 

**DISPLAY: Band Combination: IRG**

![Image](https://github.com/user-attachments/assets/5edf0ebd-85c4-4d63-9428-9156844781c0)

![Image](https://github.com/user-attachments/assets/0a0205d9-8ee5-4a19-bbae-cbffb9e8e215)


**DIsplay: RGB:**

![Image](https://github.com/user-attachments/assets/2ef9e39b-ccb9-46f8-b08f-712ae2c4e513)

![Image](https://github.com/user-attachments/assets/5a4b0520-8e28-4fb2-8751-635313ce2124)

