# DiT-SR for 4-Channel Satellite Image Super-Resolution

This repository contains an adapted version of the **Effective Diffusion Transformer Architecture for Image Super-Resolution (DiT-SR)**, specifically modified to perform super-resolution on 4-channel satellite imagery, such as SEN2NAIP data (RGB + Near-Infrared).

The core of this project involves integrating a custom 4-channel Variational Autoencoder (VAE) and a new data loading pipeline to handle the specific format of multi-spectral satellite images.

<p align="center">
  <img src="assets/framework.jpg" width="800">
  <br>
  <i>Original DiT-SR Framework. This project adapts the model for 4-channel inputs.</i>
</p>

## Key Features
- **4-Channel Image Support**: Natively processes 4-band imagery (e.g., RGB+NIR) instead of standard 3-channel RGB.
- **Custom VAE Integration**: Utilizes a 4-channel VAE for encoding and decoding multi-spectral images into the latent space.
- **SEN2NAIP Dataset Loader**: Includes a custom PyTorch dataset class for loading pairs of low-resolution and high-resolution 4-channel GeoTIFF files.
- **Device-Agnostic Training**: The training pipeline is configured to run on NVIDIA (CUDA), Apple Silicon (MPS), or CPU devices automatically.

## 1. Setup and Installation

### Prerequisites
- Python 3.10+
- Anaconda or Miniconda

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n dit_sr python=3.10 -y
    conda activate dit_sr
    ```

3.  **Install the required dependencies:**
    The `requirements.txt` file contains all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

## 2. Data Preparation (SEN2NAIP)

This project requires a specific directory structure for the training and validation data.

Create a root data folder and organize your images as follows, where each scene has its own sub-folder containing the low-resolution (`lr.tif`) and high-resolution (`hr.tif`) pairs.

```
/path/to/your/dataset/
├── train/
│   ├── scene_1/
│   │   ├── lr.tif
│   │   └── hr.tif
│   ├── scene_2/
│   │   ├── lr.tif
│   │   └── hr.tif
│   └── ...
└── val/
    ├── scene_A/
    │   ├── lr.tif
    │   └── hr.tif
    └── ...
```

The dataloader (`datapipe/sen2naip_dataset.py`) is designed to walk through these directories and automatically pair the `lr.tif` and `hr.tif` files.

## 3. Pre-trained Models

This model relies on two sets of pre-trained weights that must be downloaded and placed in the correct folders.

1.  **Original DiT-SR Checkpoints**:
    These include models for loss calculation. Download them from the original author's provided link.
    - **Download Link**: [Google Drive](https://drive.google.com/drive/folders/15EQYY3aKUKB9N3ec-AsXAZlhdCFzhT4R?usp=sharing)
    - **Action**: Place the downloaded `.pth` files into the `weights/` directory.

2.  **4-Channel VAE Checkpoint**:
    This is the custom VAE for handling 4-channel imagery.
    - **Download Link**: [**NOTE: You need to host this file and provide a link here.**]
    - **Action**: Download the `sen2naip_vae_from_opensr.ckpt` file and place it into the `checkpoints/` directory.

The directory structure should look like this after setup:
```
.
├── checkpoints/
│   └── sen2naip_vae_from_opensr.ckpt
├── weights/
│   ├── file1.pth
│   └── ...
└── ...
```

## 4. Training

The training process is managed by `torchrun` for distributed training and configured via YAML files.

- **Configuration File**: `configs/realsr_DiT.yaml`
- **Training Script**: `main.py`

To start training on your SEN2NAIP dataset, run the following command. Note that you should override the data paths in the config with command-line arguments for portability.

```bash
torchrun --standalone --nproc_per_node=8 main.py \
    --cfg_path configs/realsr_DiT.yaml \
    --save_dir /path/to/your/save_directory \
    --data_train_gt /path/to/your/dataset/train \
    --data_val_gt /path/to/your/dataset/val
```
*(Note: Adjust `--nproc_per_node` to the number of GPUs you have available.)*


## 5. Inference

To run super-resolution on new images using your trained model, use the `inference.py` script.

1.  First, ensure your trained model checkpoint (`.pth` file) is available.
2.  Run the inference command:

```bash
python inference.py --task realsr --scale 4 \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path /path/to/your/trained_model.pth \
    -i /path/to/input/images \
    -o /path/to/output/results
```

## 6. Testing the Pipeline

This repository includes scripts to help you verify your setup before launching a full training run.

### Test the Dataloader
You can test the `SEN2NAIPDataset` to ensure your data is being loaded and processed correctly.

```bash
python test_dataloader.py
```
This script will load the configuration from `configs/realsr_DiT.yaml` and attempt to fetch one batch of data, printing the tensor shapes and value ranges.

### Test for Overfitting
To perform a quick sanity check of the entire training pipeline (model, data, and loss function), you can run an overfitting test on a single batch of data. The loss should decrease steadily.

```bash
python overfit_test.py
```
This will run 300 training iterations on one batch and print the loss, confirming that the model is learning.

## Acknowledgements
- This work is built upon the original **DiT-SR** repository by Cheng et al.
- We sincerely appreciate the code release of [ResShift](https://github.com/zsyOAOA/ResShift), [DiT](https://github.com/facebookresearch/DiT), [FFTFormer](https://github.com/kkkls/FFTformer), [SwinIR](https://github.com/JingyunLiang/SwinIR), [SinSR](https://github.com/wyf0912/SinSR), and [BasicSR](https://github.com/XPixelGroup/BasicSR).
- The 4-channel VAE was adapted from the [ESAOpenSR/opensr-model](https://github.com/ESAOpenSR/opensr-model) project.

## Citation
If you use the original DiT-SR work, please consider citing:
```
@inproceedings{cheng2025effective,
  title={Effective diffusion transformer architecture for image super-resolution},
  author={Cheng, Kun and Yu, Lei and Tu, Zhijun and He, Xiao and Chen, Liyu and Guo, Yong and Zhu, Mingrui and Wang, Nannan and Gao, Xinbo and Hu, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2455--2463},
  year={2025}
}
```