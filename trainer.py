# ==============================================================================
# 1. IMPORTS
# ==============================================================================

# VAE for 4-channel imagery
from four_channel_vae.autoencoder import AutoencoderKL

# Standard library and third-party imports
import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

# Local application imports
from datapipe.datasets import create_dataset
from utils import util_net
from utils import util_common
from utils import util_image
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

# PyTorch imports
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP


# ==============================================================================
# 2. TRAINER BASE CLASS
# (This class contains your device-agnostic setup and other boilerplate)
# ==============================================================================

class TrainerBase:
    """
    A base class for handling the boilerplate of training, including distributed
    setup, seeding, logging, and checkpointing.
    """
    def __init__(self, configs):
        self.configs = configs
        self.device = None # Will be set in setup_dist
        self.setup_dist()
        self.setup_seed()

    def setup_dist(self):
        """
        Your custom device-agnostic setup for distributed training.

        This method is a key contribution. It robustly checks for available
        hardware in order of preference (NVIDIA CUDA -> Apple MPS -> CPU) and
        configures the environment accordingly. This makes your code highly
        portable.
        """
        if torch.cuda.is_available():
            # Standard CUDA setup for single or multi-GPU NVIDIA training
            self.device = torch.device("cuda")
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn')
                rank = int(os.environ['LOCAL_RANK'])
                torch.cuda.set_device(rank % num_gpus)
                dist.init_process_group(
                        timeout=datetime.timedelta(seconds=3600),
                        backend='nccl',
                        init_method='env://',
                        )
            self.num_gpus = num_gpus
            self.rank = int(os.environ.get('LOCAL_RANK', 0))
            print("NVIDIA CUDA backend is available and will be used.")

        elif torch.backends.mps.is_available():
            # Apple Silicon (M1/M2/M3) GPU support
            self.device = torch.device("mps")
            self.num_gpus = 1
            self.rank = 0
            print("Apple MPS backend is available and will be used.")

        else:
            # Fallback to CPU if no GPU is available
            self.device = torch.device("cpu")
            self.num_gpus = 1
            self.rank = 0
            print("CUDA and MPS not available. Falling back to CPU.")

    def setup_seed(self, seed=None, global_seeding=None):
        # ... (Boilerplate code for seeding, left as-is) ...
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if self.device.type == 'cuda':
            if not global_seeding:
                seed += self.rank
                torch.cuda.manual_seed(seed)
            else:
                torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        # ... (Boilerplate code for logging, left as-is) ...
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def resume_from_ckpt(self):
        # ... (Boilerplate code for resuming from checkpoint, with your device-agnostic change) ...
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)
            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            # Your Change: Load checkpoint to the correct device (CPU/CUDA/MPS)
            ckpt = torch.load(self.configs.resume, map_location=self.device)
            util_net.reload_model(self.model, ckpt['state_dict'])
            if self.device.type == 'cuda': torch.cuda.empty_cache()
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=self.device)
            if self.amp_scaler is not None and "amp_scaler" in ckpt:
                self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)
        # Your Change: Conditionally enable Automatic Mixed Precision (AMP) only for CUDA.
        use_amp = self.configs.train.use_amp and self.device.type == 'cuda'
        self.amp_scaler = amp.GradScaler() if use_amp else None

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        """Your Change: Move a batch of data to the currently active device."""
        data = {key:value.to(self.device, dtype=dtype) for key, value in data.items() if isinstance(value, torch.Tensor)}
        return data

    def build_model(self):
        # ... (Boilerplate code for building the main model, with your device-agnostic changes) ...
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.to(self.device)
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag and self.device.type == 'cuda':
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
        if self.num_gpus > 1 and self.device.type == 'cuda':
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)
        else:
            self.model = model
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).to(self.device)
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]
        self.print_model_info()

    # ... (Other boilerplate methods left as-is) ...
    def build_dataloader(self): pass
    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info(f"Number of parameters: {num_params:.2f}M")
    def validation(self): pass
    def training_step(self, data): raise NotImplementedError
    def close_logger(self):
        if self.rank == 0 and hasattr(self, 'writer'):
            self.writer.close()

# ==============================================================================
# 3. CUSTOM TRAINER IMPLEMENTATION
# ==============================================================================

class TrainerDifIR(TrainerBase):
    """
    Your custom trainer for Diffusion-based Image Restoration (DifIR).

    This class inherits from TrainerBase and implements the specific logic
    for training your 4-channel latent diffusion model.
    """
    def build_model(self):
        """
        Builds the main DiT-SR model and initializes the crucial components
        for latent diffusion: the 4-channel VAE and the LPIPS loss.
        """
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # --- Your VAE Initialization Logic ---
        # This block checks the config for a VAE, loads it, and prepares it for use.
        if self.configs.get('four_channel_autoencoder') is not None:
            if self.rank == 0:
                self.logger.info("Initializing 4-Channel AutoencoderKL...")
            ae_config = self.configs.four_channel_autoencoder
            # Instantiate your custom 4-channel VAE
            autoencoder = AutoencoderKL(
                ddconfig=ae_config.get('params', {}).get('ddconfig'),
                embed_dim=ae_config.get('params', {}).get('embed_dim')
            )
            autoencoder.to(self.device)

            # Load the pre-trained weights for the VAE
            if ae_config.ckpt_path:
                if self.rank == 0: self.logger.info(f"Restoring AE from {ae_config.ckpt_path}")
                ckpt = torch.load(ae_config.ckpt_path, map_location=self.device)
                if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
                missing, unexpected = autoencoder.load_state_dict(ckpt, strict=False)
                if self.rank == 0:
                    self.logger.info(f"AE Missing keys: {missing}")
                    self.logger.info(f"AE Unexpected keys: {unexpected}")

            # Set the VAE to be non-trainable, as it's used as a fixed feature extractor
            trainable = ae_config.get('trainable', False)
            for p in autoencoder.parameters(): p.requires_grad_(trainable)
            autoencoder.eval() if not trainable else autoencoder.train()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # --- LPIPS Loss Initialization ---
        # The LPIPS loss is often used for perceptual quality evaluation
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
        for p in self.lpips_loss.parameters(): p.requires_grad_(False)
        self.lpips_loss.eval()

        # --- Diffusion Process Initialization ---
        # This creates the diffusion process object using your `create_gaussian_diffusion` factory
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    def training_step(self, data):
        """
        Your implementation of a single training step.
        This was adapted for the overfitting test and latent diffusion.
        """
        self.optimizer.zero_grad()

        # Sample a random timestep for each item in the batch
        tt = torch.randint(0, self.base_diffusion.num_timesteps, (data['gt'].shape[0],), device=self.device)

        # Prepare model keyword arguments, including the low-quality image for conditioning
        model_kwargs = {'lq': data['lq']} if self.configs.model.params.cond_lq else None

        # Use automatic mixed precision (AMP) if enabled and on CUDA
        context = torch.cuda.amp.autocast if self.amp_scaler is not None else nullcontext
        with context():
            # This is the core call to your diffusion logic
            losses, z0_pred, z_t = self.base_diffusion.training_losses(
                self.model, data['gt'], data['lq'], tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs
            )
            # Use L1 loss for this configuration
            loss = losses['l1'].mean()

        # Backpropagation with optional AMP scaler
        if self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Detach losses from the computation graph for logging
        detached_losses = {k: v.detach() for k, v in losses.items()}
        return detached_losses, z0_pred.detach(), z_t.detach()
