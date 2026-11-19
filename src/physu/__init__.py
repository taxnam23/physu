"""physu: A modular Python physics library powered by Pint."""

from __future__ import annotations

from . import (
    constants,
    dynamics,
    electromagnetism,
    energy,
    exceptions,
    fluids,
    kinematics,
    optics,
    thermodynamics,
    units,
    validation,
    waves,
)
from .exceptions import PhysicalDomainError, PhysuError
from .units import Quantity, ureg

__version__ = "0.8.0"

__all__ = [
    "PhysicalDomainError",
    "PhysuError",
    "Quantity",
    "__version__",
    "constants",
    "dynamics",
    "electromagnetism",
    "energy",
    "exceptions",
    "fluids",
    "kinematics",
    "optics",
    "thermodynamics",
    "units",
    "ureg",
    "validation",
    "waves",
]
