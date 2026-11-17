"""Gravitational and elastic potential energy formulas with Pint unit validation."""

from __future__ import annotations

import pint

from ..constants import GRAVITATIONAL_CONSTANT, STANDARD_GRAVITY
from ..dynamics.gravitation import gravitational_force
from ..dynamics.newton import weight
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity


def gravitational_potential_energy(
    mass: QuantityLike,
    height: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute near-surface gravitational potential energy: ``U_g = m * g * h``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        height: Height relative to reference level (dimension ``[length]``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Gravitational potential energy as a ``pint.Quantity`` in ``joule``.
    """
    w = weight(mass, gravity=gravity)
    h = validate_quantity(height, "[length]", name="height")
    return to_unit(w * h, "J")


def universal_gravitational_potential_energy(
    mass_1: QuantityLike,
    mass_2: QuantityLike,
    distance: QuantityLike,
    *,
    gravitational_constant: QuantityLike = GRAVITATIONAL_CONSTANT,
) -> pint.Quantity:
    """Compute universal gravitational potential energy: ``U_G = -G * m1 * m2 / r``.

    Args:
        mass_1: Mass of first body (dimension ``[mass]``, ``>= 0``).
        mass_2: Mass of second body (dimension ``[mass]``, ``>= 0``).
        distance: Center-to-center separation (dimension ``[length]``, ``> 0``).
        gravitational_constant: Universal gravitational constant (compatible with
            ``m^3 / (kg * s^2)``, ``> 0``, defaults to ``GRAVITATIONAL_CONSTANT``).

    Returns:
        Universal gravitational potential energy as a ``pint.Quantity`` in ``joule``.
    """
    f_g = gravitational_force(
        mass_1,
        mass_2,
        distance,
        gravitational_constant=gravitational_constant,
    )
    r = validate_quantity(
        distance,
        "[length]",
        name="distance",
        allow_negative=False,
        allow_zero=False,
    )
    result = to_unit(-f_g * r, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def elastic_potential_energy(
    spring_constant: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute elastic potential energy stored in an ideal spring: ``U_e = 0.5 * k * x^2``.

    Args:
        spring_constant: Spring stiffness (compatible with ``N / m``, ``>= 0``).
        displacement: Spring displacement from equilibrium (dimension ``[length]``).

    Returns:
        Elastic potential energy as a ``pint.Quantity`` in ``joule``.
    """
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
    )
    x = validate_quantity(displacement, "[length]", name="displacement")
    return to_unit(0.5 * k * (x**2), "J")


__all__ = [
    "elastic_potential_energy",
    "gravitational_potential_energy",
    "universal_gravitational_potential_energy",
]
