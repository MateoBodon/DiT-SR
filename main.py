# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import argparse
from omegaconf import OmegaConf

# Custom utility functions
from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool


# ==============================================================================
# 2. ARGUMENT PARSER
# ==============================================================================

def get_parser(**parser_kwargs):
    """
    Sets up and returns the argument parser for command-line options.

    This function defines the core command-line arguments needed to run the
    training, such as the configuration file path and paths for saving
    and resuming experiments.
    """
    parser = argparse.ArgumentParser(**parser_kwargs)

    # Argument for specifying the directory to save checkpoints and logs
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )

    # Argument to resume training from a specific checkpoint
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="Resume from the save_dir or a specific checkpoint file (.pth)",
            )

    # Argument for the main YAML configuration file path
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/realsr_DiT.yaml",
            help="Path to the main YAML configuration file",
            )

    args = parser.parse_args()
    return args


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    This is the main entry point of the training script.
    """
    # 1. Parse command-line arguments
    args = get_parser()

    # 2. Load the base configuration from the specified YAML file
    configs = OmegaConf.load(args.cfg_path)

    # 3. Merge command-line arguments into the OmegaConf object.
    #    This allows overriding YAML settings from the command line for
    #    greater flexibility (e.g., changing the save directory).
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume']:
            configs[key] = getattr(args, key)

    # 4. Dynamically instantiate the trainer object.
    #    The `configs.trainer.target` string (e.g., "trainer.TrainerDifIR")
    #    is used to get the correct trainer class.
    trainer = get_obj_from_str(configs.trainer.target)(configs)

    # 5. Start the training process.
    trainer.train()
