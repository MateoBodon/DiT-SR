# DiT-SR for 4-Channel Satellite Image Super-Resolution

This repository contains an adapted version of the **Effective Diffusion Transformer Architecture for Image Super-Resolution (DiT-SR)**, specifically modified to perform 4x super-resolution on 4-channel satellite imagery (e.g., RGB + Near-Infrared).

The project's core contribution is the integration of a custom 4-channel Variational Autoencoder (VAE) and a new data pipeline to handle multi-spectral satellite images, enabling the powerful DiT-SR architecture to work beyond standard 3-channel RGB data.

<p align="center">
  <img src="assets/framework.jpg" width="800">
  <br>
  <i>The original DiT-SR framework. This project adapts the model and data pipeline for 4-channel inputs.</i>
</p>

## Key Features
- **4-Channel Support**: Natively processes 4-band imagery (e.g., RGB+NIR) for both model conditioning and output generation.
- **Latent Diffusion**: Operates in the latent space of a pre-trained 4-channel VAE for memory and computational efficiency.
- **Custom SEN2NAIP Dataloader**: Includes a robust PyTorch `Dataset` class for loading pairs of low-resolution and high-resolution 4-channel GeoTIFF files.
- **HPC Ready**: The training pipeline is managed by `torchrun` for easy use in distributed, multi-GPU High-Performance Computing environments.
- **Device-Agnostic**: Scripts automatically detect and use NVIDIA (CUDA), Apple Silicon (MPS), or CPU hardware.

---

## How It Works: The 4-Channel Pipeline

Adapting DiT-SR for 4-channel data required two primary modifications:

1.  **First Stage Model (VAE)**: The standard 3-channel VAE was replaced with a 4-channel `AutoencoderKL` model. This VAE is responsible for encoding the high-resolution 4-channel ground truth images into a compressed latent space. The DiT model is then trained to denoise these 4-channel latents.

2.  **Conditioning**: The DiT model is conditioned on the low-resolution (LR) input image. To make this work, the 4-channel LR image is also passed through the VAE's encoder to produce a 4-channel conditioning latent. This latent is then concatenated with the noisy latent `z_t` at each step, providing the model with the necessary information to guide the super-resolution process.

The overall data flow for a single training step is:
`HR Image (4-ch) -> VAE Encoder -> Clean Latent (4-ch) -> Add Noise -> Noisy Latent z_t (4-ch)`
`LR Image (4-ch) -> VAE Encoder -> Conditioning Latent (4-ch)`
`DiT-SR Model <- [Noisy Latent z_t, Conditioning Latent, Timestep]`
`DiT-SR Model -> Denoised Latent (4-ch)`

---

## 1. Setup and Installation

### Prerequisites
- Python 3.10+
- Anaconda or Miniconda

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n dit_sr python=3.10 -y
    conda activate dit_sr
    ```

3.  **Install PyTorch:**
    Install the appropriate version of PyTorch for your hardware (see the [official PyTorch website](https://pytorch.org/get-started/locally/)). For a typical CUDA setup:
    ```bash
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Install the remaining dependencies:**
    The cleaned `requirements.txt` file contains all other necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

## 2. Data Preparation (SEN2NAIP)

### Step 2.1: Directory Structure
This project expects a specific directory structure where each scene (Region of Interest) has its own folder containing the `lr.tif` (low-resolution) and `hr.tif` (high-resolution) image pair.

Place all your `ROI_XXXX` folders into a single root directory.

/path/to/your/dataset/
├── ROI_0001/
│   ├── lr.tif
│   └── hr.tif
├── ROI_0002/
│   ├── lr.tif
│   └── hr.tif
└── ...


### Step 2.2: Create Train/Validation Split
Run the provided utility script to automatically create a training and validation split. This script creates symbolic links to avoid duplicating data.

1.  Open the `prepare_datasplit.py` script.
2.  Set the `dataset_root` variable to point to your dataset directory (e.g., `/path/to/your/dataset/`).
3.  Run the script:
    ```bash
    python prepare_datasplit.py
    ```
After running, your data folder will have the following structure, which is what the dataloader expects:

/path/to/your/dataset/
├── train/
│   ├── ROI_0001 -> (symlink to ../ROI_0001)
│   └── ...
├── val/
│   ├── ROI_0002 -> (symlink to ../ROI_0002)
│   └── ...
├── ROI_0001/
├── ROI_0002/
└── ...


## 3. Pre-trained Weights

This project requires a pre-trained 4-channel VAE.

The VAE is not provided directly. You must extract it from the full model checkpoint provided by the [ESAOpenSR/opensr-model](https://github.com/ESAOpenSR/opensr-model) project.

1.  Download their full model checkpoint (e.g., `opensr_10m_v4_v4.ckpt`).
2.  Run the provided `extract_vae.py` script to create the VAE-only checkpoint file.
    - Open `extract_vae.py`.
    - Set `downloaded_full_ckpt` to the path of the file you just downloaded.
    - Set `extracted_vae_ckpt` to `"checkpoints/sen2naip_vae_from_opensr.ckpt"`.
    - Run the script: `python extract_vae.py`
3.  This will create the required VAE file at the correct location (`checkpoints/`). The path in the YAML config already points to this location.

## 4. Testing Your Setup

Before launching a full training run, verify that your environment, data, and models are configured correctly.

### 4.1. Test the Dataloader
This script checks if your data is being loaded and processed correctly.
```bash
python test_dataloader.py

Expected Output: The script should print the shapes and value ranges of the loaded tensors, confirming that your data is normalized between -1.0 and 1.0.

4.2. Run an Overfitting Test
This test performs a sanity check of the entire training pipeline (model, data, and loss function) by attempting to overfit on a single batch.

python overfit_test.py

Expected Output: You should see the loss value printed to the console decrease steadily over 300 iterations. This confirms the model is capable of learning.

5. Training
The training process is managed by torchrun and configured via the configs/realsr_DiT.yaml file.

5.1. Update Configuration
Open configs/realsr_DiT.yaml and update the dataroot_gt paths under the data.train and data.val sections to point to your dataset's train and val directories.

5.2. Training on a Local Machine (Single or Multi-GPU)
Use torchrun to launch the training script. Adjust --nproc_per_node to the number of GPUs you have available.

# Example for a machine with 2 GPUs
torchrun --standalone --nproc_per_node=2 main.py \
    --cfg_path configs/realsr_DiT.yaml \
    --save_dir /path/to/your/save_directory

5.3. Training on HPC with Slurm
For HPC clusters using the Slurm scheduler, you can use a submission script.

Create a file named train_hpc.sh.

Copy the following content into the file, adjusting the --nodes, --gpus, and path variables as needed.

#!/bin/bash
#SBATCH --job-name=dit-sr-4ch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Activate your conda environment
source /path/to/your/conda/etc/profile.d/conda.sh
conda activate dit_sr

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

# Launch training with torchrun
torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_TASK \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main.py \
    --cfg_path configs/realsr_DiT.yaml \
    --save_dir /path/to/your/save_directory

Submit the job to the scheduler:

sbatch train_hpc.sh

6. Inference
To run super-resolution on new images using your trained model, use the inference.py script.

python inference.py --task realsr --scale 4 \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path /path/to/your/trained_model.pth \
    -i /path/to/input/images \
    -o /path/to/output/results

Acknowledgements
This work is built upon the original DiT-SR repository by Cheng et al.

We sincerely appreciate the code release of ResShift, DiT, FFTFormer, SwinIR, SinSR, and BasicSR.

The 4-channel VAE was adapted from the ESAOpenSR/opensr-model project.

Citation
If you use the original DiT-SR work, please consider citing:

@inproceedings{cheng2025effective,
  title={Effective diffusion transformer architecture for image super-resolution},
  author={Cheng, Kun and Yu, Lei and Tu, Zhijun and He, Xiao and Chen, Liyu and Guo, Yong and Zhu, Mingrui and Wang, Nannan and Gao, Xinbo and Hu, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2455--2463},
  year={2025}
}
