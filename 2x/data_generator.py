import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms

class ComposePair:
    """Compose-like class for (lr, hr) pair transforms"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr):
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr
    

class NormalizePair:
    def __init__(self, lr_mean, lr_std, hr_mean, hr_std):
        self.lr_mean = torch.tensor(lr_mean, dtype=torch.float32).view(-1, 1, 1)
        self.lr_std = torch.tensor(lr_std, dtype=torch.float32).view(-1, 1, 1)
        self.hr_mean = torch.tensor(hr_mean, dtype=torch.float32).view(-1, 1, 1)
        self.hr_std = torch.tensor(hr_std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, lr, hr):
        lr = lr.float()
        hr = hr.float()
        lr = (lr - self.lr_mean) / self.lr_std
        hr = (hr - self.hr_mean) / self.hr_std
        return lr, hr


class SRDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_train=True,  augmentations=None):
        """
        Args:
            df: DataFrame with hr_path,lr_path
            root_dir: Root dataset directory
            transform: Optional transforms
            is_train: Whether this is training set (for augmentations)
        """
        self.df = df
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.augmentations = augmentations 
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        with rasterio.open(self.root_dir/row['hr_path']) as src:
            hr = torch.from_numpy(src.read().astype('float32')/10000.0).float()
            # hr = hr[[2, 1, 0], :, :]  # Convert BGRI to RGB
            hr = hr[[2, 1, 0, 3], :, :]  # Convert BGRI to RGBI
            
        with rasterio.open(self.root_dir/row['lr_path']) as src:
            lr = torch.from_numpy(src.read().astype('float32')/10000.0).float()
            # lr = lr[[2, 1, 0], :, :]
            lr = lr[[2, 1, 0, 3], :, :]  # Convert BGRI to RGBI

        # with rasterio.open(self.root_dir/row['hr_path']) as src:
        #     hr = torch.from_numpy(src.read().astype('float32')).float()
            
        # with rasterio.open(self.root_dir/row['lr_path']) as src:
        #     lr = torch.from_numpy(src.read().astype('float32')).float()        

        # # Convert to tensors
        # hr_tensor = hr
        # lr_tensor = lr

        # Apply augmentations only during training

        if self.transform:
           
            lr, hr = self.transform(lr, hr)
          
            
        if self.is_train and self.augmentations:
            lr, hr = self.augmentations(lr, hr)  # Spatial transforms after normalization
            
        return {'lr': lr, 'hr': hr}