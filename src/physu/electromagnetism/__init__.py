"""Electromagnetism formulas and utilities for physu."""

from __future__ import annotations

from . import circuits, electrostatics, magnetism
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
from .magnetism import (
    cyclotron_frequency,
    cyclotron_radius,
    force_per_length_parallel_wires,
    lorentz_magnetic_force,
    magnetic_field_loop_center,
    magnetic_field_solenoid,
    magnetic_field_wire,
    magnetic_force_on_wire,
)

__all__ = [
    "circuits",
    "coulomb_force",
    "cyclotron_frequency",
    "cyclotron_radius",
    "electric_field_point_charge",
    "electric_flux",
    "electric_force",
    "electric_potential_energy",
    "electric_potential_point_charge",
    "electric_power",
    "electric_work",
    "electrostatics",
    "force_per_length_parallel_wires",
    "gauss_electric_flux",
    "joule_heating",
    "lorentz_magnetic_force",
    "magnetic_field_loop_center",
    "magnetic_field_solenoid",
    "magnetic_field_wire",
    "magnetic_force_on_wire",
    "magnetism",
    "ohms_law_current",
    "ohms_law_resistance",
    "ohms_law_voltage",
    "parallel_resistance",
    "series_resistance",
    "uniform_electric_field",
    "voltage_divider",
    "wire_resistance",
]
