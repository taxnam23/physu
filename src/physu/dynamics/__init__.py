"""Classical dynamics formulas and utilities for physu."""

from __future__ import annotations

from . import friction, newton
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
    "weight",
]
