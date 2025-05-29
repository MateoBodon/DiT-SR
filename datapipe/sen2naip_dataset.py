# DiT-SR/datapipe/sen2naip_dataset.py

import torch
from torch.utils import data as data
import numpy as np
import rasterio # For reading GeoTIFFs
from rasterio.windows import Window # For partial reads if needed for cropping
import os

# Assuming data_utils.py (with linear_transform_4b) is in DiT-SR/opensr_utils/
try:
    from opensr_utils.data_utils import linear_transform_4b
except ImportError:
    # Fallback if the path is different or not set up in PYTHONPATH
    # This will require a more robust path solution in your project
    print("Warning: Could not import linear_transform_4b from opensr_utils.data_utils. Normalization might fail.")
    print("Make sure DiT-SR/opensr_utils/ is in your PYTHONPATH or adjust the import.")
    # Define a placeholder if needed for basic flow, but this is not a real solution
    def linear_transform_4b(t_input, stage="norm"):
        print(f"Warning: Using placeholder for linear_transform_4b for stage: {stage}")
        if stage == "norm":
            return (t_input / 2000.0).clamp(0,1) # Very rough placeholder, assuming input range
        return t_input


from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor


class SEN2NAIPDataset(data.Dataset):
    """
    Paired image dataset for SEN2NAIP (4-channel GeoTIFFs).

    Args:
        opt (dict): Config dict. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path to the meta_info_file.
            io_backend (dict): IO backend type and other related GT options.
            filename_tmpl (str): Template for filename. Default: '{}'.
            gt_size (int): Cropped GT patch size.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (0, 90, 180, 270 degrees).
            scale (int): Scale factor. LQ size = GT size // scale.

            phase (str): 'train' or 'val'.
            # Additional SEN2NAIP specific options can be added here if needed
    """

    def __init__(self, opt):
        super(SEN2NAIPDataset, self).__init__()
        self.opt = opt
        self.paths = []
        self.lq_paths = []
        self.gt_paths = []

        if opt.get('meta_info_file') is not None:
            logger = get_root_logger()
            logger.info(f"Loading meta_info_file: {opt['meta_info_file']}")
            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    lq_path_rel, gt_path_rel = line.strip().split(' ')
                    self.lq_paths.append(os.path.join(opt['dataroot_lq'], lq_path_rel))
                    self.gt_paths.append(os.path.join(opt['dataroot_gt'], gt_path_rel))
        else: # Scan folder
            self.gt_paths = paired_paths_from_folder([opt['dataroot_gt']], ['gt'], opt['filename_tmpl'])['gt_paths']
            self.lq_paths = paired_paths_from_folder([opt['dataroot_lq']], ['lq'], opt['filename_tmpl'])['lq_paths']

        assert len(self.gt_paths) == len(self.lq_paths), (
            f"GT and LQ paths should have the same length, but got {len(self.gt_paths)} and {len(self.lq_paths)}")


    def _load_img_rasterio(self, filepath):
        """Load a 4-channel GeoTIFF image using rasterio. Returns (C, H, W) numpy array."""
        try:
            with rasterio.open(filepath) as src:
                # SEN2NAIP typically uses B04, B03, B02, B08 for RGBNIR at 10m
                # For NAIP (HR), it's also RGBNIR.
                # Assuming the GeoTIFFs store these as the first 4 bands.
                if src.count < 4:
                    raise ValueError(f"Image {filepath} has less than 4 bands: {src.count}")
                # Read first 4 bands. Rasterio reads as (bands, height, width)
                img = src.read(list(range(1, 5))).astype(np.float32)
            return img
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f"Error loading image {filepath} with rasterio: {e}. Returning None.")
            return None


    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]

        img_lq = self._load_img_rasterio(lq_path)
        img_gt = self._load_img_rasterio(gt_path)

        if img_lq is None or img_gt is None:
            # Handle error, e.g., by returning a dummy item or raising an error
            # that can be caught by a custom collate_fn.
            # For now, let's try to return a placeholder if one is None (not ideal)
            # A better approach is to filter out problematic images beforehand.
            logger = get_root_logger()
            logger.error(f"Failed to load image pair: LQ: {lq_path}, GT: {gt_path}. Skipping.")
            # To make collate_fn work, we need to return something of the expected structure
            # This is a placeholder and should be improved with proper error handling/filtering.
            dummy_lq = torch.zeros((4, gt_size // scale, gt_size // scale), dtype=torch.float32)
            dummy_gt = torch.zeros((4, gt_size, gt_size), dtype=torch.float32)
            return {'lq': dummy_lq, 'gt': dummy_gt, 'lq_path': "error", 'gt_path': "error"}


        # Check if shapes are as expected, SEN2NAIP might already provide patches
        # Or HR is large and LR is small, and we need to do paired random crop.
        # For DiT-SR training, fixed size patches are used (e.g., gt_size=256)

        if self.opt['phase'] == 'train':
            # Paired random crop
            # img_gt and img_lq are C, H, W numpy arrays here
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # Augmentation (hflip, vflip, rot)
            # The augment function from basicsr expects a list of ndarrays (H, W, C)
            # Our images are (C, H, W) from rasterio. Transpose before and after.
            img_gt = np.transpose(img_gt, (1, 2, 0)) # C,H,W -> H,W,C
            img_lq = np.transpose(img_lq, (1, 2, 0)) # C,H,W -> H,W,C
            
            imgs_aug = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq = imgs_aug[0], imgs_aug[1]
            
            # Transpose back to (C, H, W) and convert to tensor
            img_gt_tensor = img2tensor(img_gt, bgr2rgb=False, float32=True) # bgr2rgb=False since it's not BGR
            img_lq_tensor = img2tensor(img_lq, bgr2rgb=False, float32=True)
        else: # Validation/Testing phase
            # Convert to tensor without random crop or augmentation (usually)
            # basicsr PairedImageDataset does center crop or direct conversion
            # For now, let's assume val images are already appropriately sized or handled by `img2tensor`
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32))
            img_lq_tensor = torch.from_numpy(img_lq.astype(np.float32))


        # Normalization using linear_transform_4b
        # linear_transform_4b expects BxCxHxW or CxHxW, current tensors are CxHxW
        # Add batch dim, apply, remove batch dim
        img_lq_norm = linear_transform_4b(img_lq_tensor.unsqueeze(0), stage="norm").squeeze(0)
        img_gt_norm = linear_transform_4b(img_gt_tensor.unsqueeze(0), stage="norm").squeeze(0)
        
        # The (val - 0.5) / 0.5 normalization from DiT-SR trainer.py (lines 564-565)
        # should be REMOVED if using linear_transform_4b, as this function
        # already normalizes the Sentinel-2 band values to a range suitable for the VAE
        # (typically a range around [-1, 1] or [0, 1] depending on its internal scaling).

        return {'lq': img_lq_norm, 'gt': img_gt_norm, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.gt_paths)