import yaml
import torch
from omegaconf import OmegaConf

# --- CHANGE: Import SEN2NAIPDataset directly ---
from datapipe.sen2naip_dataset import SEN2NAIPDataset

def main():
    # --- 1. Load Configuration ---
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
    print(f"LQ Path: {configs.data.train.get('dataroot_lq', 'Not specified')}") # Use .get for safety
    print("--------------------------\n")

    # --- 2. Create Dataset Directly ---
    try:
        # --- CHANGE: Instantiate the class directly, bypassing the framework ---
        print("--- Attempting to create dataset directly... ---")
        train_dataset = SEN2NAIPDataset(configs.data.train)
        
    except Exception as e:
        print(f"--- [ERROR] An exception occurred while creating the dataset ---")
        # Print the full traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return
        
    print(f"\nSUCCESS: Dataset created. Found {len(train_dataset)} image pairs.")

    # --- 3. Create DataLoader ---
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2, # Use a small batch size for testing
        shuffle=True,
        num_workers=0
    )
    print("SUCCESS: DataLoader created.\n")
    
    # --- 4. Fetch and Inspect a Batch ---
    print("--- Fetching one batch of data ---")
    try:
        batch = next(iter(dataloader))
        lq, gt = batch['lq'], batch['gt']

        print("SUCCESS: Batch fetched!")
        print(f"LQ tensor shape: {lq.shape}")
        print(f"GT tensor shape: {gt.shape}")

        # --- 5. Verify Data ---
        print("\n--- Verifying data properties ---")
        lq_min, lq_max = lq.min(), lq.max()
        gt_min, gt_max = gt.min(), gt.max()
        print(f"LQ value range: [{lq_min:.4f}, {lq_max:.4f}]")
        print(f"GT value range: [{gt_min:.4f}, {gt_max:.4f}]")
        print("\nSUCCESS: Dataloader test complete.")

    except Exception as e:
        print(f"\n--- [ERROR] An error occurred while fetching or inspecting the batch ---")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()