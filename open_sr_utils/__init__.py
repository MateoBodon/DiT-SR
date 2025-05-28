# type: ignore[attr-defined]
"""Latent diffusion model trained in RGBN optical remote sensing imagery"""

import sys
from opensr_model.srmodel import *
from opensr_model import *

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
