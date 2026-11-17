"""Fundamental physical constants as Pint quantities bound to ``physu.ureg``.

Values follow the 2019 SI redefinition and CODATA 2022 recommended values.
"""

from __future__ import annotations

import math

import pint

from .units import Quantity, to_unit

# Mechanics & gravitation
STANDARD_GRAVITY: pint.Quantity = Quantity(9.80665, "m / s^2")
GRAVITATIONAL_CONSTANT: pint.Quantity = Quantity(6.67430e-11, "m^3 / (kg * s^2)")

# Relativity & quantum mechanics
SPEED_OF_LIGHT: pint.Quantity = Quantity(299_792_458, "m / s")
PLANCK_CONSTANT: pint.Quantity = Quantity(6.62607015e-34, "J * s")
REDUCED_PLANCK_CONSTANT: pint.Quantity = Quantity(
    6.62607015e-34 / (2.0 * math.pi),
    "J * s",
)

# Electromagnetism
ELEMENTARY_CHARGE: pint.Quantity = Quantity(1.602176634e-19, "C")
VACUUM_PERMITTIVITY: pint.Quantity = Quantity(8.8541878188e-12, "F / m")
VACUUM_PERMEABILITY: pint.Quantity = Quantity(1.25663706127e-6, "N / A^2")

# Thermodynamics & chemistry
BOLTZMANN_CONSTANT: pint.Quantity = Quantity(1.380649e-23, "J / K")
AVOGADRO_CONSTANT: pint.Quantity = Quantity(6.02214076e23, "1 / mol")
MOLAR_GAS_CONSTANT: pint.Quantity = AVOGADRO_CONSTANT * BOLTZMANN_CONSTANT
STANDARD_ATMOSPHERE: pint.Quantity = Quantity(101_325, "Pa")
STEFAN_BOLTZMANN_CONSTANT: pint.Quantity = to_unit(
    (2.0 * (math.pi**5) * (BOLTZMANN_CONSTANT**4))
    / (15.0 * (PLANCK_CONSTANT**3) * (SPEED_OF_LIGHT**2)),
    "W / (m^2 * K^4)",
)

# Atomic & particle masses
ELECTRON_MASS: pint.Quantity = Quantity(9.1093837139e-31, "kg")
PROTON_MASS: pint.Quantity = Quantity(1.67262192595e-27, "kg")

__all__ = [
    "AVOGADRO_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
    "GRAVITATIONAL_CONSTANT",
    "MOLAR_GAS_CONSTANT",
    "PLANCK_CONSTANT",
    "PROTON_MASS",
    "REDUCED_PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "STANDARD_ATMOSPHERE",
    "STANDARD_GRAVITY",
    "STEFAN_BOLTZMANN_CONSTANT",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
]
