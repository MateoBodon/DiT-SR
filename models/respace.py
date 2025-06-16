import numpy as np
import torch as th

from typing import TYPE_CHECKING
from . import gaussian_diffusion as gd
from .gaussian_diffusion import ModelMeanType, ModelVarTypeDDPM as ModelVarType, LossType

if TYPE_CHECKING:
    from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps per section.

    For example, if there are 1000 timesteps and section_counts is [10, 20, 30],
    then the first 100 timesteps are strided to be 10 timesteps, the next 200
    are strided to be 20 timesteps, and the final 700 are strided to be 30
    timesteps.

    If the stride is a string, then it is treated as saving the timesteps
    every N steps (possibly with offset).

    :param num_timesteps: the number of timesteps in the original diffusion process.
    :param section_counts: a list of ints or a string containing comma-separated
                           ints, indicating the number of timesteps we want to
                           take from each section of the original diffusion process.
    """
    if isinstance(section_counts, str):
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


class SpacedDiffusion(gd.GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection of timesteps from the base diffusion process
                          to use.
    :param kwargs: the kwargs to pass to the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []

        # This line creates the base diffusion process from the parameters in your YAML
        base_diffusion = gd.GaussianDiffusion(**kwargs)

        # --- FIX: Get the number of steps from the object we just created ---
        self.original_num_steps = base_diffusion.num_timesteps

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = th.tensor(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)