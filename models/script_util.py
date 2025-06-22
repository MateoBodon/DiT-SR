# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import argparse
import inspect

# Local application imports
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


# ==============================================================================
# 2. DIFFUSION FACTORY FUNCTIONS
# ==============================================================================

def create_gaussian_diffusion(
        *,
        steps,
        model_mean_type,
        loss_type,
        schedule_name,
        schedule_kwargs,
        sf,
        kappa,
        etas_end,
        min_noise_level,
        normalize_input,
        latent_flag,
        **kwargs # Catch any other unused args
):
    """
    Your custom factory function to create the Gaussian diffusion process.

    This function is a key part of your design. It reads the diffusion
    parameters from the YAML configuration, creates the appropriate `eta`
    noise schedule, and then instantiates your custom `GaussianDiffusion`
    class with all the necessary settings.

    Args:
        (all): Parameters are passed via keyword from the YAML config file.
               See `configs/realsr_DiT.yaml` for details.

    Returns:
        gd.GaussianDiffusion: An initialized diffusion process object.
    """
    # 1. Create the noise schedule based on the specified name (e.g., 'exponential')
    #    This calls your custom `get_named_eta_schedule` function.
    sqrt_etas = gd.get_named_eta_schedule(
        schedule_name=schedule_name,
        num_diffusion_timesteps=steps,
        min_noise_level=min_noise_level,
        etas_end=etas_end,
        kappa=kappa,
        kwargs=schedule_kwargs
    )

    # 2. Instantiate and return your custom GaussianDiffusion object.
    #    This uses the `eta` schedule and other parameters to set up the
    #    diffusion mathematics for training and sampling.
    diffusion = gd.GaussianDiffusion(
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=getattr(gd.ModelMeanType, model_mean_type),
        loss_type=getattr(gd.LossType, loss_type),
        sf=sf,
        normalize_input=normalize_input,
        latent_flag=latent_flag
    )

    return diffusion


# ==============================================================================
# 3. BOILERPLATE DDPM DIFFUSION FACTORY
# (This is original code from the repository, left as-is)
# ==============================================================================

def create_gaussian_diffusion_ddpm(
    *,
    beta_start,
    beta_end,
    sf=4,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    timestep_respacing=None,
    scale_factor=1.0,
):
    """
    This is the original factory function for creating a standard DDPM-style
    diffusion process based on beta schedules. It is preserved for reference.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)

    # Note: This returns a different class, `SpacedDiffusionDDPM`, which is not
    # part of the provided files but would be the DDPM equivalent of your SpacedDiffusion.
    return SpacedDiffusionDDPM(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarTypeDDPM.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarTypeDDPM.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarTypeDDPM.LEARNED_RANGE
        ),
        scale_factor=scale_factor,
        sf=sf,
    )
