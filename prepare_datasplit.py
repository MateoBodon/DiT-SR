import os
import random
import shutil
import pathlib

def create_train_val_split(root_path, train_ratio=0.9):
    """
    Creates train and validation splits for the SEN2NAIP dataset.

    Args:
        root_path (str): The path to the directory containing all the ROI_XXXX folders.
        train_ratio (float): The proportion of data to be used for training.
    """
    root = pathlib.Path(root_path)
    if not root.is_dir():
        print(f"Error: Provided path '{root_path}' is not a directory.")
        return

    # Find all ROI directories
    roi_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('ROI_')])
    if not roi_dirs:
        print(f"Error: No 'ROI_XXXX' directories found in '{root_path}'.")
        return
        
    print(f"Found {len(roi_dirs)} total ROI directories.")

    # Shuffle for a random split
    random.shuffle(roi_dirs)

    # Determine split index
    split_index = int(len(roi_dirs) * train_ratio)
    train_rois = roi_dirs[:split_index]
    val_rois = roi_dirs[split_index:]

    print(f"Splitting into {len(train_rois)} training samples and {len(val_rois)} validation samples.")

    # Create train and val directories
    train_path = root / 'train'
    val_path = root / 'val'
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    print(f"Created directories: '{train_path}' and '{val_path}'")

    # Create symbolic links
    print("Creating symbolic links...")
    for roi in train_rois:
        os.symlink(roi, train_path / roi.name, target_is_directory=True)

    for roi in val_rois:
        os.symlink(roi, val_path / roi.name, target_is_directory=True)
        
    print("\nSUCCESS: Dataset split complete!")
    print("Your YAML should now point to the new 'train' and 'val' directories.")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Set the path to your dataset folder containing the ROI folders
    dataset_root = "/Volumes/SSD/GIS/cross-sensor"
    # ---
    
    create_train_val_split(dataset_root)