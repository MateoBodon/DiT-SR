import torch
from torch.utils import data as data
import numpy as np
import rasterio # For reading GeoTIFFs
import os

# It's good practice to ensure the utility can be found.
# This assumes open_sr_utils is in the project root.
from open_sr_utils.data_utils import linear_transform_4b

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient

class SEN2NAIPDataset(data.Dataset):
    """
    A robust dataset for loading 4-channel SEN2NAIP GeoTIFF data.
    This version uses rasterio for reliable 4-channel TIFF loading.
    """
    def __init__(self, opt):
        super(SEN2NAIPDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder, self.lq_folder = self.opt['dataroot_gt'], self.opt['dataroot_lq']
        
        # Scan for paired paths
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.opt.get('filename_tmpl', '{}')
        )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        
        # --- Load LQ and GT images using rasterio ---
        lq_path = self.paths[index]['lq_path']
        gt_path = self.paths[index]['gt_path']

        try:
            with rasterio.open(lq_path) as src:
                # rasterio reads as (channels, height, width)
                img_lq = src.read().astype(np.float32)
            with rasterio.open(gt_path) as src:
                img_gt = src.read().astype(np.float32)
        except Exception as e:
            raise IOError(f"Error opening or reading GeoTIFF file at {lq_path} or {gt_path}. Error: {e}")

        # --- Data Augmentation (for training phase) ---
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # Paired random crop. Input should be (C, H, W) numpy arrays
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            # Augment (flip, rotation). Input should be numpy arrays.
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # --- Convert to Tensor ---
        # Data is already in (C, H, W) format from rasterio, so just convert to tensor
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)).float()
        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()

        # --- Normalize Tensors to [-1, 1] ---
        img_gt = linear_transform_4b(img_gt, stage="norm")
        img_lq = linear_transform_4b(img_lq, stage="norm")

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)