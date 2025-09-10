import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms



class NormalizePair:
    def __init__(self, lr_mean, lr_std, hr_mean, hr_std):
        self.lr_mean = torch.tensor(lr_mean).view(-1,1,1)
        self.lr_std = torch.tensor(lr_std).view(-1,1,1)
        self.hr_mean = torch.tensor(hr_mean).view(-1,1,1)
        self.hr_std = torch.tensor(hr_std).view(-1,1,1)
        
    def __call__(self, lr, hr):
        lr = (lr - self.lr_mean) / self.lr_std
        hr = (hr - self.hr_mean) / self.hr_std  # Use HR stats for HR
        return lr, hr

class AllDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, augmentations=None):
        """
        Args:
            root_dir (str): Path to esrgan_dataset folder
            split (str): One of ['train', 'val', 'test']
            transform: Optional transforms
            augmentations: Optional spatial augmentations (applied after transform if training)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.augmentations = augmentations
        
        # Get all HR/LR pairs
        hr_dir = self.root_dir / split / 'hr'
        lr_dir = self.root_dir / split / 'lr'
        
        self.hr_paths = sorted(list(hr_dir.glob('*.tif')))
        self.lr_paths = sorted(list(lr_dir.glob('*.tif')))
        
        # Verify matching pairs
        assert len(self.hr_paths) == len(self.lr_paths), "Mismatched HR/LR counts"
        for hr, lr in zip(self.hr_paths, self.lr_paths):
            assert hr.stem == lr.stem, f"Mismatched filenames: {hr.stem} vs {lr.stem}"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # Load 8-channel images
        with rasterio.open(self.hr_paths[idx]) as src:
            hr = src.read().astype('float32') / 10000.0  # Normalize reflectance
            
        with rasterio.open(self.lr_paths[idx]) as src:
            lr = src.read().astype('float32') / 10000.0
            
        # Convert to tensor
        hr = torch.from_numpy(hr).float()
        lr = torch.from_numpy(lr).float()
        
        # Reorder bands: first 3 must be RGB (BGRN → RGB + other bands)
        # Assuming original order is [B2(Blue), B3(Green), B4(Red), B8(NIR), B5, B6, B7, B8A]
        hr_reordered = torch.cat([
            hr[[2, 1, 0], :, :],  # BGR → RGB (first 3 channels)
            hr[3:, :, :]           # Remaining bands in original order
        ], dim=0)
        
        lr_reordered = torch.cat([
            lr[[2, 1, 0], :, :],  # BGR → RGB
            lr[3:, :, :]
        ], dim=0)

        # Apply transforms
        if self.transform:
            lr_reordered, hr_reordered = self.transform(lr_reordered, hr_reordered)

        # Apply augmentations (only for training)
        if self.split == 'train' and self.augmentations:
            lr_reordered, hr_reordered = self.augmentations(lr_reordered, hr_reordered)

        return {
            'lr': lr_reordered,    # 8-channel with first 3 as RGB
            'hr': hr_reordered,    # 8-channel with first 3 as RGB
            'filename': self.hr_paths[idx].stem
        }