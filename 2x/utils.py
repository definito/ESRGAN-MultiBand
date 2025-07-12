import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_stats(dataset, num_bands=3, type="hr"):
    pixel_sum = np.zeros(num_bands)
    pixel_sq_sum = np.zeros(num_bands)
    pixel_count = 0
    
    for sample in tqdm(dataset, desc=f'Calculating {type.upper()} stats'):
        img = sample[type].numpy()
        pixel_sum += img.sum(axis=(1, 2))
        pixel_sq_sum += (img**2).sum(axis=(1, 2))
        pixel_count += img.shape[1] * img.shape[2]
    
    means = pixel_sum / pixel_count
    stds = np.sqrt((pixel_sq_sum / pixel_count) - means**2)
    return means.tolist(), stds.tolist()


def prepare_dataloaders(
    manifest_path,
    root_dir,
    seed=42,
    batch_size=32,
    num_workers=4,
    normalize_means_stds=None,
    load_and_split_manifest_fn=None,
    SRDataset_class=None,
    ComposePair_class=None,
    NormalizePair_class=None
):
    assert load_and_split_manifest_fn, "load_and_split_manifest_fn is required"
    assert SRDataset_class and ComposePair_class and NormalizePair_class, "Dataset and transform classes must be passed"

    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    (train_df, val_df, test_df), _ = load_and_split_manifest_fn(manifest_path, root_dir, SEED=seed)

    if normalize_means_stds is None:
        train_dataset_for_stats = SRDataset_class(train_df, root_dir, transform=None, is_train=True)
        lr_mean, lr_std = calculate_stats(train_dataset_for_stats, type="lr")
        hr_mean, hr_std = calculate_stats(train_dataset_for_stats, type="hr")
    else:
        lr_mean, lr_std, hr_mean, hr_std = normalize_means_stds

    normalize = NormalizePair_class(lr_mean, lr_std, hr_mean, hr_std)
    transform = ComposePair_class([normalize])

    train_ds = SRDataset_class(train_df, root_dir, transform=transform, is_train=True)
    val_ds = SRDataset_class(val_df, root_dir, transform=transform, is_train=False)
    test_ds = SRDataset_class(test_df, root_dir, transform=transform, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, (lr_mean, lr_std, hr_mean, hr_std)
