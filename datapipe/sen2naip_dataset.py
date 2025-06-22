# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import torch
from torch.utils import data as data
import numpy as np
import rasterio
import os
import glob
import time

# Imports from the basicsr framework and your custom utilities
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient
from open_sr_utils.data_utils import linear_transform_4b


# ==============================================================================
# 2. SEN2NAIP DATASET CLASS
# ==============================================================================

# The decorator registers this class with the basicsr framework,
# allowing it to be dynamically created from a YAML config file.
@DATASET_REGISTRY.register()
class SEN2NAIPDataset(data.Dataset):
    """
    A PyTorch Dataset class specifically designed to load and process paired
    low-resolution (LR) and high-resolution (HR) 4-channel satellite imagery
    from the SEN2NAIP dataset, stored as GeoTIFF (.tif) files.

    This class assumes a specific directory structure where each region of
    interest (ROI) is in its own subfolder, containing `lr.tif` and `hr.tif` files.
    Example:
        /path/to/data/
        ├── train/
        │   ├── scene_1/
        │   │   ├── lr.tif
        │   │   └── hr.tif
        │   └── scene_2/
        │       ├── lr.tif
        │       └── hr.tif
    """
    def __init__(self, opt):
        """
        Initializes the dataset object.

        Args:
            opt (dict): A dictionary of options, typically loaded from a YAML
                        configuration file. It contains paths and settings
                        for the dataset.
        """
        super(SEN2NAIPDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # The root directory containing the data (e.g., the 'train' or 'val' folder)
        self.data_root = self.opt['dataroot_gt']
        self.paths = []

        # Validate that the provided data path exists
        if not os.path.isdir(self.data_root):
            raise ValueError(f"Data root path is not a directory: {self.data_root}")

        # Get a sorted list of all scene directories (e.g., 'scene_1', 'scene_2')
        roi_dirs = sorted([d for d in os.listdir(self.data_root) if not d.startswith('.')])

        # Iterate through each scene directory to find the lr/hr pairs
        for roi_dir_name in roi_dirs:
            roi_path = os.path.join(self.data_root, roi_dir_name)
            if not os.path.isdir(roi_path):
                continue

            lq_path = os.path.join(roi_path, 'lr.tif')
            gt_path = os.path.join(roi_path, 'hr.tif')

            # If both lr.tif and hr.tif exist, add them as a pair to the list
            if os.path.exists(lq_path) and os.path.exists(gt_path):
                self.paths.append({'lq_path': lq_path, 'gt_path': gt_path})

        # If no pairs were found, raise an error to alert the user
        if not self.paths:
            raise ValueError(
                f"No image pairs found in {self.data_root}. Please check the path and folder structure."
            )

    def __getitem__(self, index):
        """
        Retrieves a single data sample from the dataset at the given index.

        This method performs the following steps:
        1. Reads the 4-channel LR and HR GeoTIFF images.
        2. Converts raw pixel values to reflectance (0.0 to 1.0 range).
        3. Transposes dimensions for compatibility with `basicsr` augmentations.
        4. If in 'train' phase, applies random cropping and augmentations (flips, rotations).
        5. Transposes dimensions back for PyTorch (C, H, W).
        6. Converts numpy arrays to PyTorch tensors.
        7. Normalizes tensor values to the [-1.0, 1.0] range required by the model.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the LR tensor, HR tensor, and their paths.
                  {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': str, 'gt_path': str}
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        lq_path = self.paths[index]['lq_path']
        gt_path = self.paths[index]['gt_path']

        # Use rasterio to open the 4-channel GeoTIFF files
        try:
            with rasterio.open(lq_path) as src:
                img_lq = src.read().astype(np.float32)
            with rasterio.open(gt_path) as src:
                img_gt = src.read().astype(np.float32)
        except Exception as e:
            raise IOError(f"Error opening or reading GeoTIFF file at {lq_path} or {gt_path}. Error: {e}")

        # Convert raw digital numbers to Top-of-Atmosphere (TOA) reflectance.
        # Sentinel-2 data is often scaled by 10000.
        img_lq = img_lq / 10000.0
        img_gt = img_gt / 10000.0

        # Transpose from (C, H, W) to (H, W, C) for compatibility with basicsr's
        # augmentation functions which expect the channel dim to be last.
        img_lq = img_lq.transpose(1, 2, 0).copy()
        img_gt = img_gt.transpose(1, 2, 0).copy()

        # Apply augmentations only during the training phase
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Randomly crop the HR and LR images to a consistent patch
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Apply random horizontal flips and rotations
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # Transpose back to the PyTorch standard (C, H, W) format
        img_lq = img_lq.transpose(2, 0, 1)
        img_gt = img_gt.transpose(2, 0, 1)

        # Convert numpy arrays to PyTorch tensors
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()

        # Normalize the tensor values from [0, 1] to [-1, 1] range for the model
        img_gt = linear_transform_4b(img_gt, stage="norm")
        img_lq = linear_transform_4b(img_lq, stage="norm")

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.paths)
