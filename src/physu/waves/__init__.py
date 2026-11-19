"""Oscillations, mechanical waves, acoustics, and Doppler formulas for physu."""

from __future__ import annotations

from . import mechanical, oscillations
from .mechanical import (
    spherical_wave_intensity,
    string_wave_power,
    string_wave_speed,
    wave_frequency,
    wave_intensity,
    wave_phase_speed,
    wave_speed,
    wavelength,
    wavenumber,
)
from .oscillations import (
    physical_pendulum_period,
    shm_acceleration,
    shm_displacement,
    shm_energy,
    shm_max_acceleration,
    shm_max_speed,
    shm_speed_at_position,
    shm_velocity,
    simple_pendulum_period,
    spring_mass_angular_frequency,
    spring_mass_period,
)

__all__ = [
    "mechanical",
    "oscillations",
    "physical_pendulum_period",
    "shm_acceleration",
    "shm_displacement",
    "shm_energy",
    "shm_max_acceleration",
    "shm_max_speed",
    "shm_speed_at_position",
    "shm_velocity",
    "simple_pendulum_period",
    "spherical_wave_intensity",
    "spring_mass_angular_frequency",
    "spring_mass_period",
    "string_wave_power",
    "string_wave_speed",
    "wave_frequency",
    "wave_intensity",
    "wave_phase_speed",
    "wave_speed",
    "wavelength",
    "wavenumber",
]
