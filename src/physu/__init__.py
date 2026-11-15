"""physu: A modular Python physics library powered by Pint."""

from __future__ import annotations

from . import constants, exceptions, kinematics, units, validation
from .exceptions import PhysicalDomainError, PhysuError
from .units import Quantity, ureg

__version__ = "0.1.0"

__all__ = [
    "PhysicalDomainError",
    "PhysuError",
    "Quantity",
    "__version__",
    "constants",
    "exceptions",
    "kinematics",
    "units",
    "ureg",
    "validation",
]
