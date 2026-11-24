"""Finite-crystal PET projection using parallelproj's Joseph kernels."""

from .geometry import CrystalGeometry
from .projector import CrystalProjector, TOFParameters
from .sampling import SamplingConfig

__all__ = ["CrystalGeometry", "CrystalProjector", "SamplingConfig", "TOFParameters"]
__version__ = "0.1.0"
