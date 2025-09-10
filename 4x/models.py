import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import functools

#################################################

#Improvised from: 
#https://github.com/wonbeomjang/ESRGAN-pytorch/tree/master
#https://github.com/xinntao/ESRGAN/tree/master
#https://www.geeksforgeeks.org/image-super-resolution-with-esrgan-using-pytorch/

# https://github.com/xinntao/ESRGAN/tree/master : Same Generator here to work with Torch Lightning


#################################################
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=False)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35]) # Extracting up to convBlock 5 layer(paper's feature extractor)

    def forward(self, img):
        return self.vgg19_54(img)

    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class ResidualDenseBlock(nn.Module):
    def __init__(self, filters, gc=32, bias=True, use_ca=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(filters + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(filters + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(filters + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(filters + 4 * gc, filters, 3, 1, 1, bias=bias)
        self.ca = ChannelAttention(filters) if use_ca else None
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.ca is not None:
            x5 = self.ca(x5)  # channel attention
        return x5 * 0.2 + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, gc=32, use_ca=True):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(filters, gc, use_ca=use_ca)
        self.rdb2 = ResidualDenseBlock(filters, gc, use_ca=use_ca)
        self.rdb3 = ResidualDenseBlock(filters, gc, use_ca=use_ca)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=23, num_upsample=4, gc=32, bias=True, use_ca=True):
        super(GeneratorRRDB, self).__init__()
        self.upscale = num_upsample

        RRDB_block_f = functools.partial(ResidualInResidualDenseBlock, filters=filters, gc=gc, use_ca=use_ca)

        # Feature extraction
        self.initial_conv = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1, bias=bias)
        
        # Main processing 
        self.residual_blocks = make_layer(RRDB_block_f, num_res_blocks)
        self.post_residual_conv = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=bias)

        # Upsampling
        if self.upscale in [2, 4]:
            self.upsample_conv1 = nn.Conv2d(filters, filters, 3, 1, 1, bias=bias)
        if self.upscale == 4:
            self.upsample_conv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=bias)

        # Final 
        self.high_res_conv = nn.Conv2d(filters, filters, 3, 1, 1, bias=bias)
        self.output_conv = nn.Conv2d(filters, channels, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        initial_features = self.initial_conv(x)
        
        residual_features = self.residual_blocks(initial_features)
        processed_features = self.post_residual_conv(residual_features)
        
        features = initial_features + processed_features  # Skip connection

        if self.upscale in [2, 4]:
            features = self.lrelu(self.upsample_conv1(F.interpolate(features, scale_factor=2, mode='nearest')))
        if self.upscale == 4:
            features = self.lrelu(self.upsample_conv2(F.interpolate(features, scale_factor=2, mode='nearest')))

        high_res_features = self.lrelu(self.high_res_conv(features))
        output_image = self.output_conv(high_res_features)
        
        return output_image

class Discriminator(nn.Module):
    def __init__(self, input_shape, filters=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate(filters):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)