"""Geometric and wave optics formulas and utilities for physu."""

from __future__ import annotations

from . import interference, lenses, refraction
from .interference import (
    double_slit_angle,
    double_slit_fringe_position,
    double_slit_fringe_spacing,
    michelson_displacement,
    thin_film_constructive_thickness,
)
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
    "double_slit_angle",
    "double_slit_fringe_position",
    "double_slit_fringe_spacing",
    "interference",
    "lateral_magnification",
    "lenses",
    "lensmaker_focal_length",
    "michelson_displacement",
    "mirror_focal_length",
    "optical_power",
    "refraction",
    "refractive_index",
    "snell_refraction_angle",
    "speed_in_medium",
    "thin_film_constructive_thickness",
    "thin_lens_focal_length",
    "thin_lens_image_distance",
    "wavelength_in_medium",
]
