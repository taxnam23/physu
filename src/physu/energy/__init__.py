"""Work, energy, and power formulas and utilities for physu."""

from __future__ import annotations

from . import conservation, kinetic, potential, power, work
from .conservation import (
    escape_speed,
    final_height_conservation,
    final_speed_conservation,
    mechanical_efficiency,
    mechanical_energy,
    non_conservative_work,
)
from .kinetic import (
    kinetic_energy,
    kinetic_energy_from_momentum,
    momentum_from_kinetic_energy,
    speed_from_kinetic_energy,
)
from .potential import (
    elastic_potential_energy,
    gravitational_potential_energy,
    universal_gravitational_potential_energy,
)
from .power import (
    average_power,
    force_from_power,
    instantaneous_power,
    velocity_from_power,
    work_from_power,
)
from .work import (
    elastic_work,
    gravitational_work,
    mechanical_work,
    net_work,
)

__all__ = [
    "average_power",
    "conservation",
    "elastic_potential_energy",
    "elastic_work",
    "escape_speed",
    "final_height_conservation",
    "final_speed_conservation",
    "force_from_power",
    "gravitational_potential_energy",
    "gravitational_work",
    "instantaneous_power",
    "kinetic",
    "kinetic_energy",
    "kinetic_energy_from_momentum",
    "mechanical_efficiency",
    "mechanical_energy",
    "mechanical_work",
    "momentum_from_kinetic_energy",
    "net_work",
    "non_conservative_work",
    "potential",
    "power",
    "speed_from_kinetic_energy",
    "universal_gravitational_potential_energy",
    "velocity_from_power",
    "work",
    "work_from_power",
]
