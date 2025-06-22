# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import torch
import os


# ==============================================================================
# 2. VAE EXTRACTION FUNCTION
# ==============================================================================

def extract_vae_weights(full_ckpt_path, output_vae_ckpt_path, vae_prefix="first_stage_model."):
    """
    Extracts the weights of a VAE (first stage model) from a larger,
    combined model checkpoint file.

    This is a common requirement when working with latent diffusion models,
    where a full system checkpoint contains weights for both the VAE and the
    main diffusion U-Net. This script creates a new, smaller checkpoint file
    containing only the VAE weights.

    Args:
        full_ckpt_path (str): Path to the full model checkpoint file (.ckpt).
        output_vae_ckpt_path (str): Path where the new VAE-only checkpoint will be saved.
        vae_prefix (str): The key prefix used for the VAE's layers in the full
                          checkpoint's state dictionary.
    """
    print(f"Loading full checkpoint from: {full_ckpt_path}")
    # Load the full checkpoint onto the CPU to avoid unnecessary GPU memory usage.
    checkpoint = torch.load(full_ckpt_path, map_location="cpu")

    # Checkpoints can store the model's state_dict under different keys.
    # We check for common keys like 'state_dict' or 'model'.
    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
        print("Found 'state_dict' key in checkpoint.")
    elif "model" in checkpoint:
        full_state_dict = checkpoint["model"]
        print("Found 'model' key in checkpoint.")
    else:
        # If no common key is found, assume the root of the checkpoint is the state_dict.
        full_state_dict = checkpoint
        print("Using the root of the checkpoint as state_dict.")

    vae_state_dict = {}
    found_vae_weights = False
    print(f"Attempting to extract VAE weights with prefix: '{vae_prefix}'")

    # Iterate through all keys in the full state dictionary
    for key, value in full_state_dict.items():
        # If a key starts with the specified VAE prefix, it belongs to the VAE
        if key.startswith(vae_prefix):
            # Remove the prefix to create the correct key for the standalone VAE model
            new_key = key[len(vae_prefix):]
            vae_state_dict[new_key] = value
            found_vae_weights = True

    if not found_vae_weights:
        print(f"\nWarning: No weights were found with the prefix '{vae_prefix}'.")
        print("Please manually inspect the keys in the checkpoint and update the `vae_key_prefix` variable.")
        # print("Available top-level keys in the checkpoint:", full_state_dict.keys())
        return

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_vae_ckpt_path), exist_ok=True)

    # Save the new state_dict containing only the VAE weights
    print(f"\nSaving extracted VAE weights to: {output_vae_ckpt_path}")
    torch.save(vae_state_dict, output_vae_ckpt_path)
    print("Extraction complete.")


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    # Path to the downloaded full model checkpoint that contains the VAE.
    # ADJUST THIS PATH to your downloaded file.
    downloaded_full_ckpt = "/Users/mateobodon/Downloads/opensr_10m_v4_v4.ckpt"

    # The desired output path for the new, VAE-only checkpoint file.
    # ADJUST THIS PATH to where you want to save the file.
    extracted_vae_ckpt = "/Users/mateobodon/Downloads/sen2naip_vae_from_opensr.ckpt"

    # The prefix for VAE model keys in the full checkpoint's state_dict.
    # "first_stage_model." is the standard prefix for VAEs in Stable Diffusion
    # and other LDM-based models.
    vae_key_prefix = "first_stage_model."
    # --- End Configuration ---

    if not os.path.exists(downloaded_full_ckpt):
        print(f"Error: Full checkpoint not found at {downloaded_full_ckpt}")
        print("Please download the required file and update the path in this script.")
    else:
        extract_vae_weights(downloaded_full_ckpt, extracted_vae_ckpt, vae_key_prefix)

        print("\nNext steps:")
        print("1. A new file should have been created at the `extracted_vae_ckpt` path.")
        print("2. Make sure your `realsr_DiT.yaml` config file points to this new VAE checkpoint.")
        print(f"   In YAML: ckpt_path: {extracted_vae_ckpt}")
