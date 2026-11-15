"""Kinematics formulas and utilities for physu."""

from __future__ import annotations

from . import accelerated, uniform
from .accelerated import (
    acceleration,
    displacement_from_velocities,
    final_velocity,
    time_to_displacement,
    time_to_velocity,
    torricelli_velocity,
)
from .accelerated import displacement as accelerated_displacement
from .accelerated import position as accelerated_position
from .uniform import average_velocity, displacement, position, travel_time

__all__ = [
    "accelerated",
    "accelerated_displacement",
    "accelerated_position",
    "acceleration",
    "average_velocity",
    "displacement",
    "displacement_from_velocities",
    "final_velocity",
    "position",
    "time_to_displacement",
    "time_to_velocity",
    "torricelli_velocity",
    "travel_time",
    "uniform",
]
