# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import yaml
import torch
from omegaconf import OmegaConf
import os

# Your custom Trainer and Dataset classes
from trainer import TrainerDifIR
from datapipe.sen2naip_dataset import SEN2NAIPDataset


# ==============================================================================
# 2. UTILITY & MAIN FUNCTION
# ==============================================================================

class SimpleLogger:
    """A basic logger class to print messages to the console for this test."""
    def info(self, msg):
        print(msg)

def main():
    """
    Main function to run the overfitting test.

    This script performs a critical sanity check on the entire training pipeline.
    It attempts to overfit the model on a single batch of data. If the loss
    decreases steadily, it indicates that the model, data loader, optimizer,
    and loss function are all wired together correctly and the model is capable
    of learning.
    """
    # --- Step 1: Load Configuration ---
    # Load the same YAML configuration used for the full training run.
    config_path = './configs/realsr_DiT.yaml'
    print("--- Loading Configuration ---")
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    configs = OmegaConf.create(yaml_config)

    # --- Step 2: Build the Trainer, Model, and Optimizer ---
    # We leverage the Trainer class as it conveniently handles all the complex
    # setup for the model, VAE, optimizer, and device placement.
    print("\n--- Building Trainer, Model, and Optimizer ---")
    trainer = TrainerDifIR(configs)
    trainer.logger = SimpleLogger() # Use the simple console logger for this test
    trainer.build_model()
    trainer.setup_optimizaton()

    # --- Step 3: Get a Single Batch of Data ---
    # We only need one batch to test if the model can memorize it.
    print("\n--- Preparing a Single Batch of Data for Overfitting ---")
    dataset = SEN2NAIPDataset(configs.data.train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.train.microbatch, shuffle=True)

    # Fetch one batch and move it to the correct device (CPU/CUDA/MPS)
    the_only_batch = next(iter(dataloader))
    the_only_batch = trainer.prepare_data(the_only_batch)
    print("Data batch is ready.")

    # --- Step 4: The Overfitting Loop ---
    print("\n--- Starting Overfitting Test (Loss should decrease rapidly) ---")
    trainer.model.train() # Ensure the model is in training mode

    # Repeatedly train on the exact same batch of data
    for i in range(1, 301):
        losses, _, _ = trainer.training_step(the_only_batch)

        if i % 10 == 0: # Print the loss every 10 steps to observe the trend
            print(f"Iteration {i:03d} | Loss: {losses['l1'].mean().item():.6f}")

    print("\n--- Overfitting Test Complete ---")
    print("If the loss value above steadily decreased, your model and training loop are working correctly!")


# ==============================================================================
# 3. EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # Set a dummy 'LOCAL_RANK' environment variable. This is required by the
    # Trainer's distributed setup logic, even when running on a single device.
    os.environ['LOCAL_RANK'] = '0'
    main()
