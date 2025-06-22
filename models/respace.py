# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import numpy as np
import torch as th

# Typing imports for type hinting
from typing import TYPE_CHECKING

# Local application imports
from . import gaussian_diffusion as gd
from .gaussian_diffusion import ModelMeanType, ModelVarTypeDDPM as ModelVarType, LossType

# This allows for type hinting the GaussianDiffusion class without circular imports
if TYPE_CHECKING:
    from .gaussian_diffusion import GaussianDiffusion


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps per section. This allows for non-uniform
    skipping of steps, which is key for methods like DDIM.

    For example, if there are 1000 timesteps and section_counts is [10, 20, 30],
    then the first 100 timesteps are strided to be 10 timesteps, the next 200
    are strided to be 20 timesteps, and the final 700 are strided to be 30
    timesteps.

    Args:
        num_timesteps (int): The number of timesteps in the original diffusion process.
        section_counts (str or list): A list of ints or a string of comma-separated
                                      ints indicating the number of timesteps to
                                      take from each section.

    Returns:
        set: A set of integer timesteps to use for the sampling process.
    """
    if isinstance(section_counts, str):
        # Handle DDIM-style spacing, e.g., "ddim100"
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per_section = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start = 0
    result = []
    for i, section_count in enumerate(section_counts):
        size = size_per_section + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 0:
            raise ValueError(f"cannot have non-positive section count {section_count}")
        result.extend(
            np.linspace(start, start + size - 1, num=section_count, endpoint=True, dtype=int)
        )
        start += size
    return set(result)


# ==============================================================================
# 3. SPACED DIFFUSION CLASS (Your Implementation)
# ==============================================================================

class SpacedDiffusion(gd.GaussianDiffusion):
    """
    A wrapper for a GaussianDiffusion process that allows for skipping steps.
    This is essential for faster sampling techniques like DDIM.

    This class takes a set of `use_timesteps` from the original diffusion
    process and re-calculates a new, shorter set of betas corresponding to
    this "respaced" schedule.
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []

        # This line creates the base diffusion process from the parameters in your YAML.
        # It's important that this uses your new `GaussianDiffusion` class, not the DDPM one.
        base_diffusion = gd.GaussianDiffusion(**kwargs)

        # YOUR FIX: Store the original number of timesteps from the base diffusion
        # process. This is needed by the _WrappedModel to correctly scale the timesteps.
        self.original_num_steps = base_diffusion.num_timesteps

        # Recalculate betas for the new, shorter schedule
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # Pass the newly calculated betas to the parent GaussianDiffusion constructor
        kwargs["betas"] = th.tensor(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        # Wrap the model to handle timestep re-mapping before calling the parent method
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        # Also wrap the model during training loss calculation
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        """
        Wraps the U-Net model to manage the mapping from the new, shorter
        timestep schedule to the original, full-length schedule.
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is handled by the _WrappedModel, so this becomes a no-op.
        return t


class _WrappedModel:
    """
    A helper class that wraps the U-Net model. Its main job is to intercept
    the timestep tensor `ts`, map the "spaced" timesteps (e.g., 0 to 99) back
    to their original values (e.g., 0, 10, 20...), and then pass them to the
    actual model.
    """
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        # `ts` are the timesteps from the spaced schedule (e.g., 0, 1, 2...)
        # We use `timestep_map` to find the corresponding original timesteps
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        # If required, rescale the timesteps to the standard [0, 1000] range
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        # Call the actual U-Net model with the re-mapped and rescaled timesteps
        return self.model(x, new_ts, **kwargs)
