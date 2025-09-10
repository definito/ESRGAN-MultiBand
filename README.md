# ESRGAN-MultiBand 

**ESRGAN-MultiBand** is a PyTorch Lightning-based implementation of the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN), adapted for multispectral satellite imagery. Unlike typical RGB applications, this version supports 4+ spectral bands (e.g., near-infrared, red-edge) commonly used in remote sensing datasets.

>  This repository is currently being updated. Finalized documentation, pretrained models, and all training scripts will be added soon.


# SEN2-VENUS ESRGAN Super-Resolution (2x and 4x)
This repository implements a complete PyTorch Lightning pipeline for super-resolution on the SEN12-VENUS dataset, using an enhanced ESRGAN architecture with 4-band (RGB+IR) input.

**_Dataset Tested:_ DIV2k(RGB- small subset)_2x , 2x_sen2venus RGB only, 2x_sen2vusRGBI,  4x sen2venus**

| Dataset           | Input Bands | Epochs | PSNR (dB) | SSIM  | LPIPS-1 | LPIPS-2 |
|-------------------|-------------|--------|-----------|-------|---------|---------|
| DIV2K-2x (subset) | RGB         | 200    | 30.48     | 0.869 | –       | –       |
| Sen2Venus-2x      | RGBI        | 200    | 37.45     | 0.950 | 0.085 (RGB) | 0.152 (NIRG) |
| Sen2Venus-4x      | 5678        | 200    | 37.51     | 0.925 | 0.177 (567) | 0.154 (678) |



# Key Features:
2× Super-Resolution using a GAN-based architecture with RRDB generator and VGG19-based perceptual loss.

4-channel support: RGB + Infrared input/output.

Built-in dataloader with rasterio support for GeoTIFFs.

Pixel, Content, GAN, and SSIM Losses for balanced training.

Lightning-ready training loop with checkpointing, logging, and visual validation image saving.

Easy-to-use runner with custom normalization, reproducible split, and visualization.


### **DIV2K:** 

![Image](https://github.com/user-attachments/assets/7658cdd1-80d4-47f9-9563-b04cc914382d)

![Image](https://github.com/user-attachments/assets/4cd3072b-8ca4-401c-afd5-e6125d73ec68)


# Result for 2x SR
### **SEN2VENUS_2x - band- RBGI** 

**DISPLAY: Band Combination: IRG**
<img width="2098" height="723" alt="esrgan" src="https://github.com/user-attachments/assets/209336c5-c67e-40e9-a6ac-7e16987516ac" />




**DIsplay: RGB:**
<img width="2096" height="721" alt="esrgan" src="https://github.com/user-attachments/assets/28482221-440b-451f-b293-0a4b5e6292af" />



# Result for 4x SR
### **SEN2VENUS_2x - band- 5678** 
**DIsplay: Band 5-6-7:**
<img width="2109" height="725" alt="esrgan" src="https://github.com/user-attachments/assets/51c94456-bd9c-4e34-9349-96a09f0d200c" />

**DIsplay: Band 8-7-6:**
<img width="2670" height="916" alt="image" src="https://github.com/user-attachments/assets/2f7a9af9-d7a5-42cc-9a81-a4eb36169091" />



