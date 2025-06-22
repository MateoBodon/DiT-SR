# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import os
import random
import shutil
import pathlib


# ==============================================================================
# 2. DATA SPLITTING FUNCTION
# ==============================================================================

def create_train_val_split(root_path, train_ratio=0.9):
    """
    Creates train and validation splits for the SEN2NAIP dataset by creating
    symbolic links (symlinks) to the original data folders.

    This script avoids duplicating data. It scans a root directory for scene
    folders (e.g., 'ROI_XXXX'), shuffles them, and then creates 'train' and
    'val' subdirectories containing symlinks to the original scenes.

    Args:
        root_path (str): The path to the directory containing all the ROI_XXXX folders.
        train_ratio (float): The proportion of the total data to allocate for training.
    """
    root = pathlib.Path(root_path)
    if not root.is_dir():
        print(f"Error: Provided path '{root_path}' is not a directory.")
        return

    # Find all directories that start with 'ROI_'
    roi_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('ROI_')])
    if not roi_dirs:
        print(f"Error: No 'ROI_XXXX' directories found in '{root_path}'.")
        return

    print(f"Found {len(roi_dirs)} total ROI directories.")

    # Shuffle the list of directories to ensure a random train/validation split
    random.shuffle(roi_dirs)

    # Calculate the split point based on the training ratio
    split_index = int(len(roi_dirs) * train_ratio)
    train_rois = roi_dirs[:split_index]
    val_rois = roi_dirs[split_index:]

    print(f"Splitting into {len(train_rois)} training samples and {len(val_rois)} validation samples.")

    # Create the 'train' and 'val' directories inside the root path
    train_path = root / 'train'
    val_path = root / 'val'
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    print(f"Created directories: '{train_path}' and '{val_path}'")

    # Create symbolic links for the training set
    print("Creating symbolic links for training set...")
    for roi in train_rois:
        # Create a symlink from the original ROI directory to the new 'train' directory
        link_path = train_path / roi.name
        if not link_path.exists(): # Avoid errors if the link already exists
            os.symlink(roi, link_path, target_is_directory=True)

    # Create symbolic links for the validation set
    print("Creating symbolic links for validation set...")
    for roi in val_rois:
        # Create a symlink from the original ROI directory to the new 'val' directory
        link_path = val_path / roi.name
        if not link_path.exists():
            os.symlink(roi, val_path / roi.name, target_is_directory=True)

    print("\nSUCCESS: Dataset split complete!")
    print("Your YAML configuration should now point to the new 'train' and 'val' directories.")


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --- IMPORTANT ---
    # This is the main configuration variable for this script.
    # Set this path to the root folder of your dataset, which contains
    # all the 'ROI_XXXX' scene folders.
    dataset_root = "/Volumes/SSD/GIS/cross-sensor"
    # ---

    print(f"Preparing to split dataset at: {dataset_root}")
    create_train_val_split(dataset_root)

