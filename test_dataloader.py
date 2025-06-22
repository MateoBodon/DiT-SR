# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import yaml
import torch
from omegaconf import OmegaConf
import traceback

# Your custom Dataset class for 4-channel data
from datapipe.sen2naip_dataset import SEN2NAIPDataset


# ==============================================================================
# 2. MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function to test the SEN2NAIPDataset and DataLoader.

    This script is a crucial debugging tool. It isolates the data loading part
    of the pipeline to verify that:
    1. The dataset can be initialized without errors.
    2. It can correctly find and load the 4-channel GeoTIFF image pairs.
    3. It produces tensors of the correct shape and data type.
    4. The normalization is working as expected (values are in the [-1, 1] range).
    """
    # --- Step 1: Load Configuration ---
    # We load the main training config to get the dataset parameters.
    config_path = './configs/realsr_DiT.yaml'
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    configs = OmegaConf.create(yaml_config)
    print("--- Configuration Loaded ---")
    print(f"GT Path: {configs.data.train.dataroot_gt}")
    print("--------------------------\n")

    # --- Step 2: Create Dataset Directly ---
    # We instantiate the dataset class directly to test it in isolation from
    # the main trainer framework.
    try:
        print("--- Attempting to create dataset directly... ---")
        train_dataset = SEN2NAIPDataset(configs.data.train)
    except Exception as e:
        print(f"--- [ERROR] An exception occurred while creating the dataset ---")
        traceback.print_exc()
        return

    print(f"\nSUCCESS: Dataset created. Found {len(train_dataset)} image pairs.")

    # --- Step 3: Create DataLoader ---
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2, # Use a small batch size for testing
        shuffle=True,
        num_workers=0 # Use 0 workers for easier debugging
    )
    print("SUCCESS: DataLoader created.\n")

    # --- Step 4: Fetch and Inspect a Batch ---
    print("--- Fetching one batch of data ---")
    try:
        batch = next(iter(dataloader))
        lq, gt = batch['lq'], batch['gt']

        print("SUCCESS: Batch fetched!")
        print(f"LQ tensor shape: {lq.shape}")
        print(f"GT tensor shape: {gt.shape}")

        # --- Step 5: Verify Data Properties ---
        # Check the value range to ensure normalization to [-1, 1] is working.
        print("\n--- Verifying data properties ---")
        lq_min, lq_max = lq.min(), lq.max()
        gt_min, gt_max = gt.min(), gt.max()
        print(f"LQ value range: [{lq_min:.4f}, {lq_max:.4f}]")
        print(f"GT value range: [{gt_min:.4f}, {gt_max:.4f}]")
        print("\nSUCCESS: Dataloader test complete.")

    except Exception as e:
        print(f"\n--- [ERROR] An error occurred while fetching or inspecting the batch ---")
        traceback.print_exc()


# ==============================================================================
# 3. EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    main()
