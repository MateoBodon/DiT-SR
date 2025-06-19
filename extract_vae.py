import torch
import os

def extract_vae_weights(full_ckpt_path, output_vae_ckpt_path, vae_prefix="first_stage_model."):
    """
    Extracts VAE weights from a full model checkpoint.

    Args:
        full_ckpt_path (str): Path to the downloaded full model checkpoint.
        output_vae_ckpt_path (str): Path where the extracted VAE weights will be saved.
        vae_prefix (str): The prefix used for VAE model keys in the full checkpoint.
                        Commonly "first_stage_model." for LDM-based models.
    """
    print(f"Loading full checkpoint from: {full_ckpt_path}")
    # Load the full checkpoint, move to CPU to avoid GPU memory issues if not needed for this script
    checkpoint = torch.load(full_ckpt_path, map_location="cpu")

    # The actual model weights might be under a specific key, e.g., 'state_dict'
    # Adjust this based on how the checkpoint is structured.
    # If the top-level dict is the state_dict, then state_dict = checkpoint
    if "state_dict" in checkpoint:
        full_state_dict = checkpoint["state_dict"]
        print("Found 'state_dict' key in checkpoint.")
    elif "model" in checkpoint: # Another common key
        full_state_dict = checkpoint["model"]
        print("Found 'model' key in checkpoint.")
    else:
        full_state_dict = checkpoint # Assuming the checkpoint root is the state_dict
        print("Using the root of the checkpoint as state_dict.")

    vae_state_dict = {}
    found_vae_weights = False
    print(f"Attempting to extract VAE weights with prefix: '{vae_prefix}'")

    for key, value in full_state_dict.items():
        if key.startswith(vae_prefix):
            # Strip the prefix to match the standalone VAE model's key names
            new_key = key[len(vae_prefix):]
            vae_state_dict[new_key] = value
            found_vae_weights = True
            # print(f"  Extracted: {key} -> {new_key}") # Uncomment for debugging

    if not found_vae_weights:
        print(f"Warning: No weights found with the prefix '{vae_prefix}'.")
        print("Please check the checkpoint structure and the vae_prefix.")
        print("Available top-level keys in full_state_dict:", full_state_dict.keys())
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_vae_ckpt_path), exist_ok=True)

    print(f"Saving extracted VAE weights to: {output_vae_ckpt_path}")
    torch.save(vae_state_dict, output_vae_ckpt_path)
    print("Extraction complete.")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to the downloaded full SR model checkpoint
    downloaded_full_ckpt = "/Users/mateobodon/Downloads/opensr_10m_v4_v4.ckpt" # ADJUST THIS PATH

    # Desired path for the new VAE-only checkpoint
    extracted_vae_ckpt = "/Users/mateobodon/Downloads/sen2naip_vae_from_opensr.ckpt" # ADJUST THIS PATH

    # Prefix for the VAE model in the full checkpoint.
    # Based on your information, "first_stage_model." is highly likely for LDM-derived VAEs.
    # If this doesn't work, you might need to inspect the keys of the downloaded checkpoint manually.
    vae_key_prefix = "first_stage_model."
    # --- End Configuration ---

    if not os.path.exists(downloaded_full_ckpt):
        print(f"Error: Full checkpoint not found at {downloaded_full_ckpt}")
        print("Please download it first and update the path in this script.")
    else:
        extract_vae_weights(downloaded_full_ckpt, extracted_vae_ckpt, vae_key_prefix)

        print("\nNext steps:")
        print(f"1. Verify that '{extracted_vae_ckpt}' has been created.")
        print(f"   ckpt_path: {extracted_vae_ckpt}")
