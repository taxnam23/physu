"""Geometric and wave optics formulas and utilities for physu."""

from __future__ import annotations

from . import refraction
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
    "critical_angle",
    "refraction",
    "refractive_index",
    "snell_refraction_angle",
    "speed_in_medium",
    "wavelength_in_medium",
]
