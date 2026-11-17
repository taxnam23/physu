"""physu: A modular Python physics library powered by Pint."""

from __future__ import annotations

from . import constants, dynamics, energy, exceptions, kinematics, thermodynamics, units, validation
from .exceptions import PhysicalDomainError, PhysuError
from .units import Quantity, ureg

__version__ = "0.4.0"

__all__ = [
    "PhysicalDomainError",
    "PhysuError",
    "Quantity",
    "__version__",
    "constants",
    "dynamics",
    "energy",
    "exceptions",
    "kinematics",
    "thermodynamics",
    "units",
    "ureg",
    "validation",
]
