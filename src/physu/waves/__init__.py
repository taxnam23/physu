"""Oscillations, mechanical waves, acoustics, and Doppler formulas for physu."""

from __future__ import annotations

from . import oscillations
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
    "spring_mass_angular_frequency",
    "spring_mass_period",
]
