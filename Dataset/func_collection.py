# Main Libraries
import os
import pandas as pd
import zipfile
import traceback
import rasterio
from tqdm import tqdm
import numpy as np
import requests
import shutil
import math
# Table Libraries
from tabulate import tabulate
import re

from glob import glob

# Ploting Libraries
from PIL import Image
import matplotlib.pyplot as plt

# DataSet Libraries
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
from rasterio.shutil import copy as rio_copy
from sklearn.model_selection import train_test_split
from rasterio.warp import reproject, Resampling
import random

def download_zip(url, output_path):
    """
    Downloads a ZIP file from a URL and saves it to the specified path.

    Args:
        url (str): The URL to the ZIP file.
        output_path (str): The file path to save the ZIP (e.g., 'file.zip').
    """
    response = requests.get(url, stream=True)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f" ZIP Saved to - {output_path}")
    else:
        print(f" Failed to download ZIP- {response.status_code}")


def main_zip_extract(zip_file_path:str, output_dir:str):
    """
    Extracts a zip file to the specified output directory.
    Args:
        zip_file_path (str): Path to the zip file.
        output_dir (str): Directory where the contents of the zip file will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)



def data_extract(input_dir:str, input_csv_path:str, output_dir_name:str, output_csv_path:str, input_index_col=False, input_dataframe_sep='\t', output_dataframe_sep=',',output_index=False):   
    """
    Extracts data from a CSV file and a zip file, and saves the result to a new CSV file.
    Args:
        input_dir (str): The directory containing the zip files.
        input_csv_path (str): The path to the input CSV file.
        output_dir_name (str): The name of the output directory.
        output_csv_path (str): The path to the output CSV file.
        input_index_col (bool): Whether to use the first column as the index.
        input_dataframe_sep (str): The separator used in the input CSV file.
        output_dataframe_sep (str): The separator used in the output CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path, index_col=input_index_col, sep=input_dataframe_sep)
    #Drop the first column
    # df = df.drop(df.columns[0], axis=1)
    if df.columns[0].startswith("Unnamed") or pd.to_numeric(df[df.columns[0]], errors="coerce").notna().all():
        df = df.drop(columns=[df.columns[0]])
    # check df size
    main_index_size = df.size
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the rows of the DataFrame
    row_mapping = {}
    for col in df.columns:
        target_dir = os.path.join(output_dir, col)
        os.makedirs(target_dir, exist_ok=True)
        # Iterate through the coloums of the DataFrame
        # for index, cell in enumerate(df[col]):
        for index, cell in tqdm(enumerate(df[col]), desc=f"Processing {col}", total=len(df[col])):
            # Check if the cell contains a zip file path
            if pd.isna(cell) or "/" not in str(cell):
                continue
            try:

                # zipname.zip/path/inside/zip.tif -> zipname.zip, path/inside/zip.tif
                zip_name, inner_path = cell.split("/", 1)
                zip_path = os.path.join(input_dir, zip_name)


                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract the inner path
                    if inner_path not in zip_ref.namelist():
                        print(f"Extracting {inner_path} from {zip_name} Failed")
                    else:
                        # print(f"Extracting {inner_path} from {zip_name}")
                        with zip_ref.open(inner_path) as tif_file:
                            data = tif_file.read()

                        # Save the extracted data to a new file: just in the folder: col
                        output_file_path = os.path.join(target_dir, os.path.basename(inner_path))
                        output_file_relative_path = os.path.join(col, os.path.basename(inner_path))
                        
                        with open(output_file_path, 'wb') as output_file:
                            output_file.write(data)
                          
                        # Register in row_mapping
                        if index not in row_mapping:
                            row_mapping[index] = {}
                        row_mapping[index][col] = output_file_relative_path
                        # print(f" Extracted: {inner_path} -> {output_file_relative_path}")


            except Exception as e:
                print(f"Error: {e}")
                continue
            
    df_final = pd.DataFrame.from_dict(row_mapping, orient='index')
    df_final = df_final.sort_index()  # Keep original row order
    df_final_size = df_final.size
    df_final.to_csv(output_csv_path, sep=output_dataframe_sep, index=output_index)
    print(" Saved CSV file:", output_csv_path)
    print(f'The new CSV file is separated by {output_dataframe_sep} instead of tabs.')

    print(f'The original CSV file has {main_index_size} elements.')
    print(f'The new CSV file has {df_final_size} elements.')
    if main_index_size == df_final_size:
        print("The new CSV file has the same number of elements as the original CSV file.")
    else:
        print("The new CSV file has a different number of elements than the original CSV file.")
    


    
def data_sanity_check(data_root: str, input_csv_path: str, sep=","):
    """
    Performs a sanity check on the data in the CSV file.
    Args:
        data_root (str): The root directory where the data is stored.
        input_csv_path (str): The path to the input CSV file.
        sep (str): The separator used in the CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path, sep=sep, index_col=False)
    counter = 0
    data_folders =set()
    for i, row in df.iterrows():
        folder_list = row.tolist()
        for j in range(len(folder_list)):
            fileName = os.path.join(data_root, folder_list[j])
            folder = os.path.dirname(fileName)
            data_folders.add(folder)
            if os.path.isfile(fileName):
                counter += 1
            else:
                raise FileNotFoundError(f" File not found: {fileName}")  

    print(f"Data root: {data_root}")
    print(f"Data folders: {data_folders}")
    print(f"Total files found: {counter}")                       
  
    return True

def process_and_stack_images(csv_path, data_root, output_root, site_name):
    """
    Process Sentinel-2/VENµS images and organize with structure:
    8_channel/{site_name}/{resolution}/{sitename}_{i}.tif
    
    Args:
        csv_path (str): Path to index.csv
        data_root (str): Root directory containing image folders
        output_root (str): Base directory to save organized stacks
        site_name (str): Name of the current site being processed
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {site_name}"):
        try:
            # ===== 20m Processing =====
            # Downsample 10m to 20m
            b10m_path = os.path.join(data_root, row['b2b3b4b8_10m'])
            with rasterio.open(b10m_path) as src:
                b10m = src.read()
                profile = src.profile
            
            # Downsample by factor of 2 (10m->20m)
            downsampled = np.zeros((b10m.shape[0], b10m.shape[1]//2, b10m.shape[2]//2))
            for i in range(b10m.shape[0]):
                reproject(
                    b10m[i], downsampled[i],
                    src_transform=profile['transform'],
                    dst_transform=rasterio.Affine(
                        profile['transform'].a * 2, 0, profile['transform'].c,
                        0, profile['transform'].e * 2, profile['transform'].f
                    ),
                    src_crs=profile['crs'],
                    dst_crs=profile['crs'],
                    resampling=Resampling.average
                )
            
            # Load 20m bands
            b20m_path = os.path.join(data_root, row['b5b6b7b8a_20m'])
            with rasterio.open(b20m_path) as src:
                b20m = src.read()
            
            # Stack (10m downsampled + 20m)
            stacked_20m = np.concatenate([downsampled, b20m], axis=0)
            
            # Save 20m stack with correct naming
            os.makedirs(os.path.join(output_root, "8_channel", site_name, "20m"), exist_ok=True)
            output_path = os.path.join(output_root, "8_channel", site_name, "20m", f"{site_name}_{idx}.tif")
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=stacked_20m.shape[1],
                width=stacked_20m.shape[2],
                count=stacked_20m.shape[0],
                dtype=stacked_20m.dtype,
                crs=profile['crs'],
                transform=rasterio.Affine(
                    profile['transform'].a * 2, 0, profile['transform'].c,
                    0, profile['transform'].e * 2, profile['transform'].f
                )
            ) as dst:
                dst.write(stacked_20m)
            
            # ===== 5m Processing =====
            # Stack original 5m bands
            b5m_vis = os.path.join(data_root, row['b2b3b4b8_05m'])
            b5m_re = os.path.join(data_root, row['b5b6b7b8a_05m'])
            
            with rasterio.open(b5m_vis) as src:
                vis = src.read()
                profile_5m = src.profile
            
            with rasterio.open(b5m_re) as src:
                re = src.read()
            
            stacked_5m = np.concatenate([vis, re], axis=0)
            
            # Save 5m stack with correct naming
            os.makedirs(os.path.join(output_root, "8_channel", site_name, "5m"), exist_ok=True)
            output_path = os.path.join(output_root, "8_channel", site_name, "5m", f"{site_name}_{idx}.tif")
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=stacked_5m.shape[1],
                width=stacked_5m.shape[2],
                count=stacked_5m.shape[0],
                dtype=stacked_5m.dtype,
                crs=profile_5m['crs'],
                transform=profile_5m['transform']
            ) as dst:
                dst.write(stacked_5m)
                
        except Exception as e:
            print(f"Error processing row {idx} (site {site_name}): {e}")
            continue

    print(f"Processing complete for {site_name}. Stacks saved to {output_root}/8_channel/{site_name}")

def build_esrgan_pair(site, csv_path, data_root, output_root, base_path, lr_col, hr_col):
    """
    For each row in CSV:
      - copy LR -> {output_root}/{base_path}/{site}/LR/{site}_{i}.tif
      - copy HR -> {output_root}/{base_path}/{site}/HR/{site}_{i}.tif
    """
    df = pd.read_csv(csv_path)

    out_lr = os.path.join(output_root, base_path, site, "LR")
    out_hr = os.path.join(output_root, base_path, site, "HR")
    os.makedirs(out_lr, exist_ok=True)
    os.makedirs(out_hr, exist_ok=True)

    paired = skipped = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"[{site}]"):
        lr_src = os.path.join(data_root, str(row[lr_col]))
        hr_src = os.path.join(data_root, str(row[hr_col]))

        if not (os.path.exists(lr_src) and os.path.exists(hr_src)):
            skipped += 1
            continue

        name = f"{site}_{i}.tif"
        shutil.copyfile(lr_src, os.path.join(out_lr, name))
        shutil.copyfile(hr_src, os.path.join(out_hr, name))
        paired += 1

    print(f"[{site}] Done. Paired: {paired} | Skipped: {skipped}")

def prepare_esrgan_dataset(base_path, sites, output_root="esrgan_data", lr_dirname="20m", hr_dirname="5m"):
    """
    Reorganizes stacked images into ESRGAN-friendly structure:
    {output_root}/
    ├── train/
    │   ├── hr/  (5m images)
    │   └── lr/  (20m images/10m )
    ├── val/
    │   ├── hr/
    │   └── lr/
    └── test/
        ├── hr/
        └── lr/
    
    Args:
        base_path (str): Path to stacked_images/8_channel
        sites (list): List of site names
        output_root (str): Main output folder name
    """
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for res in ['hr', 'lr']:
            os.makedirs(os.path.join(output_root, split, res), exist_ok=True)
    
    # Process each site
    for site in sites:
        site_path = os.path.join(base_path, site)
        
        # Get all 5m (hr) and 20m (lr) images
        hr_images = sorted([f for f in os.listdir(os.path.join(site_path, hr_dirname)) if f.endswith('.tif')])
        lr_images = sorted([f for f in os.listdir(os.path.join(site_path, lr_dirname)) if f.endswith('.tif')])

        # Verify matching pairs
        assert len(hr_images) == len(lr_images), f"Mismatched counts in {site}"
        assert all(hr == lr for hr, lr in zip(hr_images, lr_images)), "Filename mismatch in {site}"
        
        # Split indices (65% train, 15% val, 15% test)
        indices = np.arange(len(hr_images))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        
        # Copy files to new structure
        for idx, img_name in enumerate(hr_images):
            src_hr = os.path.join(site_path, hr_dirname, img_name)
            src_lr = os.path.join(site_path, lr_dirname, img_name)

            if idx in train_idx:
                split = 'train'
            elif idx in val_idx:
                split = 'val'
            else:
                split = 'test'
            
            # Copy with site prefix to avoid name collisions
            dest_hr = os.path.join(output_root, split, 'hr', f"{site}_{img_name}")
            dest_lr = os.path.join(output_root, split, 'lr', f"{site}_{img_name}")
            
            shutil.copy2(src_hr, dest_hr)
            shutil.copy2(src_lr, dest_lr)
    
    print(f"ESRGAN dataset prepared at {output_root}")
    print("Split counts:")
    for split in splits:
        hr_count = len(os.listdir(os.path.join(output_root, split, 'hr')))
        lr_count = len(os.listdir(os.path.join(output_root, split, 'lr')))
        print(f"{split}: {hr_count} HR images, {lr_count} LR images")

# keep your contrast stretch
def apply_contrast_stretch(img):
    """Apply percentile-based contrast stretching to [0,1] reflectance image."""
    stretched = np.zeros_like(img)
    for i in range(img.shape[0]):
        band = img[i]
        valid = ~np.isnan(band)
        if not np.any(valid):
            stretched[i] = band
            continue
        p2, p98 = np.percentile(band[valid], (2, 98))
        if p98 > p2:
            stretched[i] = np.clip((band - p2) / (p98 - p2 + 1e-5), 0, 1)
        else:
            stretched[i] = band
    return stretched



def visualize_split_pairs(
    split_dir,
    band_combination="2x",   # "2x" -> b2b3b4b8 ; "4x" -> b5b6b7b8a
    rows_to_plot=2,
    dpi=140,
    pick="first",            # "first" or "random"
    recursive=False          # set True if LR/HR have nested site folders
):
    """
    Visualize LR/HR pairs inside a split folder (e.g., Dataset_5678/train).
    Expects:
        split_dir/
          LR/*.tif[f]
          HR/*.tif[f]
    Files are matched by identical filename (no path), e.g. ALSACE_123.tif.

    band_combination:
      - "2x": show RGB as (B4,B3,B2) -> bands [3,2,1] from (B2,B3,B4,B8)
      - "4x": show RGB as (B7,B6,B5) -> bands [3,2,1] from (B5,B6,B7,B8A)
    """
    lr_dir = os.path.join(split_dir, "lr")
    hr_dir = os.path.join(split_dir, "hr")
    if not (os.path.isdir(lr_dir) and os.path.isdir(hr_dir)):
        print(f"[ERR] Expected LR/ and HR/ inside: {split_dir}")
        return

    pat = "**/*.tif*" if recursive else "*.tif*"
    lr_files = glob(os.path.join(lr_dir, pat), recursive=recursive)
    hr_files = glob(os.path.join(hr_dir, pat), recursive=recursive)

    # map by filename (without directories)
    def by_name(files):
        d = {}
        for fp in files:
            name = os.path.basename(fp)
            d[name] = fp
        return d

    lr_map = by_name(lr_files)
    hr_map = by_name(hr_files)
    common = sorted(set(lr_map.keys()) & set(hr_map.keys()))

    if not common:
        print(f"[WARN] No common filenames found between LR and HR in {split_dir}")
        print(f"LR count: {len(lr_files)}, HR count: {len(hr_files)}")
        return

    # choose which to plot
    if pick == "random" and len(common) > rows_to_plot:
        names = random.sample(common, rows_to_plot)
    else:
        names = common[:rows_to_plot]

    # band index config (1-based in rasterio -> we’ll convert to 0-based when slicing)
    if str(band_combination).lower() == "2x":
        rgb_idx = [3, 2, 1]  # B4,B3,B2 from (B2,B3,B4,B8)
        title_hr = "HR (expected 5m, B4/B3/B2)"
        title_lr = "LR (expected 10m, B4/B3/B2)"
    else:  # "4x"
        rgb_idx = [3, 2, 1]  # B7,B6,B5 from (B5,B6,B7,B8A)
        title_hr = "HR (expected 5m, B7/B6/B5)"
        title_lr = "LR (expected 20m, B7/B6/B5)"

    for name in names:
        hr_path = hr_map[name]
        lr_path = lr_map[name]

        plt.figure(figsize=(12, 6), dpi=dpi)

        # HR
        try:
            with rasterio.open(hr_path) as src:
                img = src.read(rgb_idx).astype("float32") / 10000.0
                img = apply_contrast_stretch(img).transpose(1, 2, 0)
                plt.subplot(1, 2, 1)
                plt.imshow(np.clip(img, 0, 1))
                plt.title(f"{title_hr}\n{os.path.basename(hr_path)}\n{img.shape}")
                plt.axis("off")
        except Exception as e:
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title(f"HR READ ERROR\n{os.path.basename(hr_path)}")
            print(f"[HR ERROR] {hr_path}: {e}")

        # LR
        try:
            with rasterio.open(lr_path) as src:
                img = src.read(rgb_idx).astype("float32") / 10000.0
                img = apply_contrast_stretch(img).transpose(1, 2, 0)
                plt.subplot(1, 2, 2)
                plt.imshow(np.clip(img, 0, 1))
                plt.title(f"{title_lr}\n{os.path.basename(lr_path)}\n{img.shape}")
                plt.axis("off")
        except Exception as e:
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title(f"LR READ ERROR\n{os.path.basename(lr_path)}")
            print(f"[LR ERROR] {lr_path}: {e}")

        plt.suptitle(f"Split: {os.path.basename(split_dir)} | Pair: {name} | Mode: {band_combination.upper()}", y=0.98)
        plt.tight_layout()
        plt.show()

def plot_8_channel(stacked_dir, output_dir=None, rows_to_plot=3):
    """
    Plot RGB and 5-6-8 band combinations from stacked images.
    
    Args:
        stacked_dir (str): Directory containing stacked images (either 20m or 5m)
        output_dir (str): Optional directory to save plots
        rows_to_plot (int): Number of random images to plot
    """
    # Get list of stacked images
    stacked_images = list(Path(stacked_dir).glob("*.tif"))
    if not stacked_images:
        print(f"No TIFF images found in {stacked_dir}")
        return
    
    # Select random subset
    np.random.seed(42)
    sample_images = np.random.choice(stacked_images, min(rows_to_plot, len(stacked_images)), replace=False)
    
    for img_path in sample_images:
        with rasterio.open(img_path) as src:
            # Read all bands and normalize
            img = src.read().astype('float32') / 10000.0
            profile = src.profile
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Band Combinations - {img_path.name}\n{img.shape[1]}x{img.shape[2]} pixels", y=1.02)
            
            # ===== RGB Visualization (Bands 3-2-1) =====
            rgb_bands = [3, 2, 1]  # Assuming original order: [b2,b3,b4,b8,b5,b6,b7,b8a]
            if img.shape[0] >= 3:  # Check if we have enough bands
                rgb = np.stack([img[b-1] for b in rgb_bands], axis=-1)  # -1 for 0-based index
                
                # Apply contrast stretching
                p2, p98 = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
                rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0, 1)
                
                ax1.imshow(rgb_stretched)
                ax1.set_title(f"RGB (Bands {rgb_bands[0]}-{rgb_bands[1]}-{rgb_bands[2]})")
            else:
                ax1.text(0.5, 0.5, "Not enough bands for RGB", ha='center', va='center')
            
            # ===== 5-6-8 Visualization (Vegetation Analysis) =====
            veg_bands = [5, 6, 8]  # Using band positions in the STACKED image
            if img.shape[0] >= max(veg_bands):
                veg = np.stack([img[b-1] for b in veg_bands], axis=-1)
                
                # Stretch each band independently
                for i in range(3):
                    band = veg[:,:,i]
                    p2, p98 = np.percentile(band[~np.isnan(band)], (2, 98))
                    if p98 > p2:
                        veg[:,:,i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                
                ax2.imshow(veg)
                ax2.set_title(f"Vegetation (Bands {veg_bands[0]}-{veg_bands[1]}-{veg_bands[2]})")
            else:
                ax2.text(0.5, 0.5, "Not enough bands for vegetation combo", ha='center', va='center')
            
            # Save if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"bands_{img_path.stem}.png"), 
                            bbox_inches='tight', dpi=150)
            
            plt.tight_layout()
            plt.show()