import yaml
import torch
from omegaconf import OmegaConf
from datapipe.datasets import create_dataset

def main():
    # --- 1. Load Configuration ---
    config_path = './configs/realsr_DiT.yaml'
    try:
        with open(config_path, 'r') as f:
            # Using yaml.safe_load is safer
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        print("Please ensure the path is correct and you have updated the dataroot paths inside.")
        return

    configs = OmegaConf.create(yaml_config)
    
    print("--- Configuration Loaded ---")
    # Make sure you have updated these paths in your realsr_DiT.yaml file!
    print(f"GT Path: {configs.data.train.dataroot_gt}")
    print(f"LQ Path: {configs.data.train.dataroot_lq}")
    print("--------------------------\n")

    # --- 2. Create Dataset ---
    try:
        train_dataset = create_dataset(configs.data.train)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
        
    print(f"Successfully created dataset. Number of training samples: {len(train_dataset)}")

    # --- 3. Create DataLoader ---
    # Use a small batch size for testing
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0 # Use 0 workers for this simple test
    )
    print("Successfully created DataLoader.\n")
    
    # --- 4. Fetch and Inspect a Batch ---
    print("--- Fetching one batch of data ---")
    try:
        batch = next(iter(dataloader))
        lq, gt = batch['lq'], batch['gt']

        print("Batch fetched successfully!")
        print(f"LQ tensor shape: {lq.shape}")
        print(f"GT tensor shape: {gt.shape}")

        # --- 5. Verify Data ---
        print("\n--- Verifying data properties ---")
        print(f"LQ data type: {lq.dtype}")
        print(f"GT data type: {gt.dtype}")

        # Check value range for normalization. Should be close to [-1.0, 1.0]
        lq_min, lq_max = lq.min(), lq.max()
        gt_min, gt_max = gt.min(), gt.max()

        print(f"LQ value range: [{lq_min:.4f}, {lq_max:.4f}]")
        print(f"GT value range: [{gt_min:.4f}, {gt_max:.4f}]")

        if lq_min < -1.01 or lq_max > 1.01 or gt_min < -1.01 or gt_max > 1.01:
            print("\nWARNING: Data might not be correctly normalized to [-1, 1].")
        else:
            print("\nSUCCESS: Data appears to be correctly shaped and normalized.")

    except Exception as e:
        print(f"\nAn error occurred while fetching or inspecting the batch: {e}")
        print("This could be due to an issue with file loading, transformations, or data paths.")

if __name__ == '__main__':
    main()