import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_split_manifest(csv_path, root_dir, val_size=0.2, test_size=0.2, SEED=42):
    """
    Load manifest and split into train/val/test sets

    Args:
        csv_path: Path to CSV with hr_path,lr_path columns
        root_dir: Root directory of the dataset
        val_size: Validation set proportion
        test_size: Test set proportion
        SEED: Random seed for reproducibility

    Returns:
        (train_df, val_df, test_df), (train_meta, val_meta, test_meta)
    """
    df = pd.read_csv(csv_path)
    df = df[['hr_path', 'lr_path']].copy()

    # Verify required columns
    if 'hr_path' not in df.columns or 'lr_path' not in df.columns:
        raise ValueError("CSV must contain 'hr_path' and 'lr_path' columns")

    # Check file existence
    df['hr_exists'] = df['hr_path'].apply(lambda x: os.path.exists(Path(root_dir) / x))
    df['lr_exists'] = df['lr_path'].apply(lambda x: os.path.exists(Path(root_dir) / x))

    # Filter for valid pairs
    valid_df = df[df['hr_exists'] & df['lr_exists']].copy()

    # Extract site for optional stratification
    valid_df['site'] = valid_df['hr_path'].str.split('_').str[1]

    # Split into train/temp
    train_meta, temp_meta = train_test_split(
        valid_df,
        test_size=val_size + test_size,
        random_state=SEED,
        stratify=valid_df['site'] if valid_df['site'].nunique() > 1 else None
    )

    # Split temp into val/test
    val_meta, test_meta = train_test_split(
        temp_meta,
        test_size=test_size / (val_size + test_size),
        random_state=SEED,
        stratify=temp_meta['site'] if temp_meta['site'].nunique() > 1 else None
    )

    # Keep only core columns for model input
    keep_cols = ['hr_path', 'lr_path']
    train_df = train_meta[keep_cols].copy()
    val_df = val_meta[keep_cols].copy()
    test_df = test_meta[keep_cols].copy()

    return (train_df, val_df, test_df), (train_meta, val_meta, test_meta)
