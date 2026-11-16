"""Classical dynamics formulas and utilities for physu."""

from __future__ import annotations

from . import elasticity, friction, newton
from .elasticity import (
    parallel_spring_constant,
    series_spring_constant,
    spring_constant,
    spring_displacement,
    spring_force,
    spring_force_magnitude,
)
from .friction import (
    critical_incline_angle,
    friction_coefficient,
    kinetic_friction,
    max_static_friction,
    normal_force,
)
from .newton import (
    acceleration,
    force,
    inclined_normal_force,
    inclined_parallel_force,
    mass,
    weight,
)

__all__ = [
    "acceleration",
    "critical_incline_angle",
    "elasticity",
    "force",
    "friction",
    "friction_coefficient",
    "inclined_normal_force",
    "inclined_parallel_force",
    "kinetic_friction",
    "mass",
    "max_static_friction",
    "newton",
    "normal_force",
    "parallel_spring_constant",
    "series_spring_constant",
    "spring_constant",
    "spring_displacement",
    "spring_force",
    "spring_force_magnitude",
    "weight",
]
