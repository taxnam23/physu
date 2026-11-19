"""Geometric and wave optics formulas and utilities for physu."""

from __future__ import annotations

from . import lenses, refraction
from .lenses import (
    combined_optical_power,
    lateral_magnification,
    lensmaker_focal_length,
    mirror_focal_length,
    optical_power,
    thin_lens_focal_length,
    thin_lens_image_distance,
)
from .refraction import (
    brewster_angle,
    critical_angle,
    refractive_index,
    snell_refraction_angle,
    speed_in_medium,
    wavelength_in_medium,
)

__all__ = [
    "brewster_angle",
    "combined_optical_power",
    "critical_angle",
    "lateral_magnification",
    "lenses",
    "lensmaker_focal_length",
    "mirror_focal_length",
    "optical_power",
    "refraction",
    "refractive_index",
    "snell_refraction_angle",
    "speed_in_medium",
    "thin_lens_focal_length",
    "thin_lens_image_distance",
    "wavelength_in_medium",
]
