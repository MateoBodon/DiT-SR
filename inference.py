import argparse
import torch
import tifffile  # <-- CHANGE: For reading/writing 4-channel TIFFs
import numpy as np  # <-- CHANGE: For data manipulation
from pathlib import Path
from omegaconf import OmegaConf
from sampler import Sampler

# --- Argument Parser (with VAE path added) ---
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        elif v.lower() in ("no", "false", "f", "n", "0"): return False
        else: raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, required=True, help="Input path for a single 4-channel LR TIFF file.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output directory.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the main DiT-SR model config file (e.g., realsr_DiT.yaml).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained DiT-SR model checkpoint (.pth).")
    # <-- CHANGE: Added a dedicated argument for your 4-channel VAE checkpoint
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the pretrained 4-channel VAE checkpoint (.ckpt).")
    parser.add_argument("--chop_size", type=int, default=256, help="Chopping size for inference to manage memory.")
    parser.add_argument("--chop_stride", type=int, default=224, help="Chopping stride.")
    parser.add_argument("--fp32", type=str2bool, const=True, default=False, nargs="?", help="Disable AMP for inference.")
    
    args = parser.parse_args()
    return args

# --- Configuration Setup ---
def get_configs(args):
    """Loads the YAML config and correctly populates it with model and VAE paths."""
    configs = OmegaConf.load(args.config_path)
    
    # <-- CHANGE: Set model and VAE paths from command-line arguments
    configs.model.ckpt_path = args.ckpt_path
    configs.diffusion.params.sf = args.scale
    
    # This part is crucial: it sets the path for your custom 4-channel VAE
    if hasattr(configs, 'four_channel_autoencoder'):
        configs.four_channel_autoencoder.ckpt_path = args.vae_ckpt_path
    else:
        # If the key doesn't exist, create it to avoid errors
        configs['four_channel_autoencoder'] = {'ckpt_path': args.vae_ckpt_path}

    # Ensure output directory exists
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    
    return configs

# --- Custom 4-Channel Image Loader ---
def load_4channel_image(image_path):
    """
    Loads a 4-channel TIFF image, normalizes it, and prepares it for the model.
    """
    # <-- CHANGE: Use tifffile to load the image
    img = tifffile.imread(image_path)  # Shape: (H, W, 4)
    
    # Normalize from [0, 255] to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    
    # Convert to PyTorch tensor and add batch dimension: (H, W, 4) -> (1, 4, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def main():
    args = get_parser()
    configs = get_configs(args)

    # The sampler will now be initialized with the correct config pointing to your VAE
    sampler = Sampler(
                configs,
                sf=args.scale,
                chop_size=args.chop_size,
                chop_stride=args.chop_stride,
                use_amp=not args.fp32,
                seed=args.seed,
            )
            
    # <-- CHANGE: Complete inference pipeline for a single 4-channel image
    print(f"Processing input file: {args.in_path}")
    
    # 1. Load the low-resolution 4-channel image
    lr_image_tensor = load_4channel_image(args.in_path)

    # 2. Run the diffusion sampler to get the super-resolved tensor
    # The sampler's __call__ method handles moving data to the device
    sr_tensor = sampler(lr_image_tensor) # Output tensor shape: (1, 4, H_sr, W_sr)

    # 3. Post-process and save the output
    sr_image = sr_tensor.squeeze(0).permute(1, 2, 0)  # (4, H, W) -> (H, W, 4)
    sr_image = (sr_image.clamp(-1, 1) + 1) / 2 * 255   # Denormalize to [0, 255]
    sr_image = sr_image.cpu().numpy().astype(np.uint8)

    # 4. Save as a 4-channel TIFF file
    input_filename = Path(args.in_path).stem
    output_filepath = Path(args.out_path) / f"{input_filename}_x{args.scale}_DiT-SR.tif"
    tifffile.imwrite(output_filepath, sr_image)
    
    print(f"Successfully saved super-resolved image to: {output_filepath}")


if __name__ == '__main__':
    main()