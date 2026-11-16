"""Kinematics formulas and utilities for physu."""

from __future__ import annotations

from . import accelerated, circular, projectile, uniform, vertical
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
from .circular import (
    angular_displacement,
    angular_velocity_from_frequency,
    angular_velocity_from_period,
    angular_velocity_from_speed,
    arc_length,
    centripetal_acceleration,
    frequency_from_angular_velocity,
    frequency_from_period,
    period_from_angular_velocity,
    period_from_frequency,
    tangential_speed,
)
from .projectile import (
    projectile_flight_time,
    projectile_max_height,
    projectile_position,
    projectile_range,
    projectile_velocity,
    velocity_components,
)
from .uniform import average_velocity, displacement, position, travel_time
from .vertical import (
    free_fall_distance,
    free_fall_speed,
    free_fall_time,
    max_height,
    time_to_height,
    time_to_peak,
    vertical_position,
    vertical_velocity,
)

__all__ = [
    "accelerated",
    "accelerated_displacement",
    "accelerated_position",
    "acceleration",
    "angular_displacement",
    "angular_velocity_from_frequency",
    "angular_velocity_from_period",
    "angular_velocity_from_speed",
    "arc_length",
    "average_velocity",
    "centripetal_acceleration",
    "circular",
    "displacement",
    "displacement_from_velocities",
    "final_velocity",
    "free_fall_distance",
    "free_fall_speed",
    "free_fall_time",
    "frequency_from_angular_velocity",
    "frequency_from_period",
    "max_height",
    "period_from_angular_velocity",
    "period_from_frequency",
    "position",
    "projectile",
    "projectile_flight_time",
    "projectile_max_height",
    "projectile_position",
    "projectile_range",
    "projectile_velocity",
    "tangential_speed",
    "time_to_displacement",
    "time_to_height",
    "time_to_peak",
    "time_to_velocity",
    "torricelli_velocity",
    "travel_time",
    "uniform",
    "velocity_components",
    "vertical",
    "vertical_position",
    "vertical_velocity",
]
