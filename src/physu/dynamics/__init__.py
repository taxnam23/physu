"""Classical dynamics formulas and utilities for physu."""

from __future__ import annotations

from . import elasticity, friction, gravitation, momentum, newton
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
from .gravitation import (
    centripetal_force,
    gravitational_field_strength,
    gravitational_force,
    orbital_speed,
)
from .momentum import (
    average_force,
    collision_velocities,
    elastic_collision_velocities,
    impulse,
    inelastic_collision_velocity,
    linear_momentum,
    momentum_change,
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
    "average_force",
    "centripetal_force",
    "collision_velocities",
    "critical_incline_angle",
    "elastic_collision_velocities",
    "elasticity",
    "force",
    "friction",
    "friction_coefficient",
    "gravitation",
    "gravitational_field_strength",
    "gravitational_force",
    "impulse",
    "inclined_normal_force",
    "inclined_parallel_force",
    "inelastic_collision_velocity",
    "kinetic_friction",
    "linear_momentum",
    "mass",
    "max_static_friction",
    "momentum",
    "momentum_change",
    "newton",
    "normal_force",
    "orbital_speed",
    "parallel_spring_constant",
    "series_spring_constant",
    "spring_constant",
    "spring_displacement",
    "spring_force",
    "spring_force_magnitude",
    "weight",
]
