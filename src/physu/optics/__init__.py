"""Geometric and wave optics formulas and utilities for physu."""

from __future__ import annotations

from . import diffraction, interference, lenses, polarization, refraction
from .diffraction import (
    bragg_diffraction_angle,
    diffraction_grating_angle,
    grating_spacing_from_density,
    rayleigh_angular_resolution,
    single_slit_minimum_angle,
)
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
from .polarization import (
    malus_intensity,
    polarization_degree,
    polarizer_stack_intensity,
    unpolarized_polarizer_intensity,
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
    "bragg_diffraction_angle",
    "brewster_angle",
    "combined_optical_power",
    "critical_angle",
    "diffraction",
    "diffraction_grating_angle",
    "double_slit_angle",
    "double_slit_fringe_position",
    "double_slit_fringe_spacing",
    "grating_spacing_from_density",
    "interference",
    "lateral_magnification",
    "lenses",
    "lensmaker_focal_length",
    "malus_intensity",
    "michelson_displacement",
    "mirror_focal_length",
    "optical_power",
    "polarization",
    "polarization_degree",
    "polarizer_stack_intensity",
    "rayleigh_angular_resolution",
    "refraction",
    "refractive_index",
    "single_slit_minimum_angle",
    "snell_refraction_angle",
    "speed_in_medium",
    "thin_film_constructive_thickness",
    "thin_lens_focal_length",
    "thin_lens_image_distance",
    "unpolarized_polarizer_intensity",
    "wavelength_in_medium",
]
