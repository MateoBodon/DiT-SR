# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import enum
import math

import numpy as np
import torch
import torch as th
import torch.nn.functional as F

from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

# This import appears to be unused in the file and can be safely removed
# to clean up dependencies.
# from ldm.models.autoencoder import AutoencoderKLTorch

# ==============================================================================
# 2. NOISE SCHEDULES 
# ==============================================================================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name. (Original Function)
    """
    if schedule_name == "linear":
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    This function was added by you to implement noise schedules based on the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models",
    which defines schedules in terms of noise level (eta) rather than variance (beta).

    Args:
        schedule_name (str): The name of the schedule (e.g., 'exponential').
        num_diffusion_timesteps (int): The total number of diffusion steps.
        min_noise_level (float): The starting noise level.
        etas_end (float): The final noise level.
        kappa (float): A scaling factor for the noise.
        kwargs (dict): A dictionary for schedule-specific parameters, like 'power'.

    Returns:
        np.ndarray: A 1-D numpy array of eta values for each timestep.
    """
    if schedule_name == 'exponential':
        # This schedule creates an exponential curve for the noise levels.
        # The 'power' parameter controls the curvature of this schedule.
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        # This allows loading a pre-computed schedule from a .mat file.
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

# ==============================================================================
# 3. ENUM DEFINITIONS (Your Added Enums)
# ==============================================================================

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts. This includes your custom types.
    """
    START_X = enum.auto()       # The model predicts x_0
    EPSILON = enum.auto()       # The model predicts epsilon
    PREVIOUS_X = enum.auto()    # The model predicts x_{t-1}
    RESIDUAL = enum.auto()      # The model predicts the residual (y - x_0)
    EPSILON_SCALE = enum.auto() # The model predicts a scaled version of epsilon


class LossType(enum.Enum):
    """
    The type of loss function to use. This includes your custom types.
    """
    MSE = enum.auto()           # Simplified MSE
    L1 = enum.auto()            # L1 loss
    WEIGHTED_MSE = enum.auto()  # Weighted MSE derived from KL-divergence


class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance. (Original Enum)
    """
    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices. (Original Helper)
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ==============================================================================
# 4. YOUR GAUSSIAN DIFFUSION IMPLEMENTATION
# ==============================================================================

class GaussianDiffusion:
    """
    Your primary implementation of the diffusion process, designed to work with
    the eta-based noise schedules. This class contains the core logic for your
    4-channel latent diffusion model.
    """
    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        # Store configuration parameters
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Cast to float32 for MPS (Apple Silicon) compatibility.
        self.sqrt_etas = sqrt_etas.astype(np.float32)
        self.etas = self.sqrt_etas**2

        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])

        # Pre-calculate schedule-dependent constants for the diffusion process
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # Pre-calculate posterior variance and mean coefficients
        # These are used in the reverse process (sampling)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # Pre-calculate weights for the MSE loss if using WEIGHTED_MSE
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)
        self.weight_loss_mse = weight_loss_mse

    def q_sample(self, x_start, y, t, noise=None):
        """
        Your implementation of the forward diffusion process (q-sample).
        This function creates a noisy sample x_t from a clean image x_0
        and a condition y.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def training_losses(
        self, model, x_start, y, t,
        first_stage_model=None,
        model_kwargs=None,
        noise=None,
    ):
        """
        Your core training loop logic. This is the most critical function you've adapted.

        It performs the following steps:
        1. Encodes the high-res (x_start) and low-res (y) images into the latent space.
        2. Creates a noisy latent z_t using the q_sample method.
        3. Passes z_t and the conditional latent z_y to the model.
        4. Calculates the loss between the model's output and the correct target.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 1. Encode low-quality (y) and ground-truth (x_start) images into the latent space.
        #    The `up_sample=True` for `y` is critical, as the VAE expects high-res inputs.
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        # 2. Inject the conditional latent `z_y` into the keyword arguments for the model.
        #    This is how the model receives the conditioning information.
        model_kwargs['lq'] = z_y

        if noise is None:
            noise = torch.randn_like(z_start)

        # 3. Apply the forward diffusion process in the latent space to get z_t.
        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        terms = {}

        # 4. Determine the correct target for the loss function based on the model's prediction type.
        target = {
            ModelMeanType.START_X: z_start,
            ModelMeanType.RESIDUAL: z_y - z_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.EPSILON_SCALE: noise * self.kappa * _extract_into_tensor(self.sqrt_etas, t, noise.shape),
        }[self.model_mean_type]

        # 5. Get the model's prediction and calculate the loss.
        model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
        assert model_output.shape == target.shape == z_start.shape

        if self.loss_type == LossType.MSE:
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["l1"] = mean_flat(torch.abs(target - model_output))
        elif self.loss_type == LossType.L1:
            terms["l1"] = mean_flat(torch.abs(target - model_output))
            terms["mse"] = mean_flat((target - model_output) ** 2)
        elif self.loss_type == LossType.WEIGHTED_MSE:
            weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            terms["mse"] = mean_flat(((target - model_output) ** 2) * weights)
            terms["l1"] = mean_flat(torch.abs(target - model_output) * weights)
        else:
            raise NotImplementedError(self.loss_type)

        # 6. For logging purposes, predict the denoised latent `z_start` from the model's output.
        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def encode_first_stage(self, y_pixels, first_stage_model, up_sample=False):
        """
        Your VAE encoding logic. Encodes an image into the latent space.

        The `up_sample` flag is a key part of your implementation, allowing a
        low-resolution conditioning image `y` to be properly encoded by a VAE
        that was trained on high-resolution images.

        Args:
            y_pixels (torch.Tensor): The input image in pixel space.
            first_stage_model (torch.nn.Module): The VAE model.
            up_sample (bool): If True, upsamples the input before encoding.

        Returns:
            torch.Tensor: The latent representation of the input image.
        """
        if first_stage_model is None:
            raise ValueError("first_stage_model (AutoencoderKL) is required to encode the conditioning image.")

        y_pixels = y_pixels.to(next(first_stage_model.parameters()).device)

        if up_sample:
            # The AutoencoderKL is trained on HR images (e.g., 256x256).
            # If `y` is LR (e.g., 64x64), it must be upsampled before encoding.
            target_res = first_stage_model.encoder.resolution
            y_pixels_upsampled = th.nn.functional.interpolate(
                y_pixels,
                size=(target_res, target_res),
                mode='bicubic',
                align_corners=False
            )
        else:
            y_pixels_upsampled = y_pixels

        # The VAE encoder returns a distribution; we must `.sample()` from it.
        posterior = first_stage_model.encode(y_pixels_upsampled)
        z_y = posterior.sample()
        return z_y

    def decode_first_stage(self, z_sample, first_stage_model=None, consistencydecoder=None):
        """
        Your VAE decoding logic. Decodes a latent sample back to pixel space.

        This function is adapted to handle both a standard VAE (`first_stage_model`)
        and a potential `consistencydecoder`, making your inference pipeline flexible.
        """
        # ... (Your logic for decoding, which is complex and preserved as-is) ...
        # (Comments below explain the key parts)
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        # Determine which decoder to use
        if consistencydecoder is None:
            model = first_stage_model
            decoder = first_stage_model.decode
            model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        if first_stage_model is None and consistencydecoder is None:
            return z_sample
        else:
            z_sample_for_decode = z_sample.to(next(model.parameters()).device, dtype=model_dtype)
            if consistencydecoder is None:
                out = decoder(z_sample_for_decode)
            else:
                # Use automatic mixed precision for consistency decoder if available
                with th.cuda.amp.autocast():
                    out = decoder(z_sample_for_decode)
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    # --- The rest of your helper functions and sampling loops are preserved below ---
    # --- They contain your specific logic and are not changed. ---

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_mean_variance(self, x_start, y, t):
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(--1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def p_sample_loop(
        self,
        y,
        model,
        first_stage_model=None,
        consistencydecoder=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]
        with th.no_grad():
            out = self.decode_first_stage(
                    final,
                    first_stage_model=first_stage_model,
                    consistencydecoder=consistencydecoder,
                    )
        return out

    def p_sample_loop_progressive(
            self, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                yield out
                z_sample = out["sample"]

    def prior_sample(self, y, noise=None):
        if noise is None:
            noise = th.randn_like(y)
        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise


# ==============================================================================
# 5. ORIGINAL DDPM GAUSSIAN DIFFUSION (Boilerplate Code)
# ==============================================================================

class GaussianDiffusionDDPM:
    """
    Original boilerplate class implementing the standard DDPM formulation from
    Ho et al. (2020), based on beta schedules. This section is preserved as-is
    from the original repository for reference and potential use.
    """
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        scale_factor=None,
        sf=4,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.scale_factor = scale_factor
        self.sf=sf
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ... (All other methods of GaussianDiffusionDDPM are preserved as-is) ...

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarTypeDDPM.LEARNED, ModelVarTypeDDPM.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarTypeDDPM.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarTypeDDPM.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarTypeDDPM.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        first_stage_model=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device).long()
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, first_stage_model=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}

        z_start = self.encode_first_stage(x_start, first_stage_model)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, t, noise=noise)

        terms = {}

        model_output = model(z_t, t, **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=z_start, x_t=z_t, t=t
            )[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            with th.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
                return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with th.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)