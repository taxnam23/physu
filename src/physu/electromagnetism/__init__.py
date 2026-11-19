"""Electromagnetism formulas and utilities for physu."""

from __future__ import annotations

from . import electrostatics
from .electrostatics import (
    coulomb_force,
    electric_field_point_charge,
    electric_flux,
    electric_force,
    electric_potential_energy,
    electric_potential_point_charge,
    electric_work,
    gauss_electric_flux,
    uniform_electric_field,
)

__all__ = [
    "coulomb_force",
    "electric_field_point_charge",
    "electric_flux",
    "electric_force",
    "electric_potential_energy",
    "electric_potential_point_charge",
    "electric_work",
    "electrostatics",
    "gauss_electric_flux",
    "uniform_electric_field",
]
