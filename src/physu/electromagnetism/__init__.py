"""Electromagnetism formulas and utilities for physu."""

from __future__ import annotations

from . import circuits, electrostatics
from .circuits import (
    electric_power,
    joule_heating,
    ohms_law_current,
    ohms_law_resistance,
    ohms_law_voltage,
    parallel_resistance,
    series_resistance,
    voltage_divider,
    wire_resistance,
)
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
    "circuits",
    "coulomb_force",
    "electric_field_point_charge",
    "electric_flux",
    "electric_force",
    "electric_potential_energy",
    "electric_potential_point_charge",
    "electric_power",
    "electric_work",
    "electrostatics",
    "gauss_electric_flux",
    "joule_heating",
    "ohms_law_current",
    "ohms_law_resistance",
    "ohms_law_voltage",
    "parallel_resistance",
    "series_resistance",
    "uniform_electric_field",
    "voltage_divider",
    "wire_resistance",
]
