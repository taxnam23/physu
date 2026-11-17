"""Translational kinetic energy formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity


def _validate_energy_and_mass(
    kinetic_energy: QuantityLike,
    mass: QuantityLike,
    *,
    allow_zero_mass: bool,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Validate non-negative kinetic energy and mass domain constraints."""
    k = validate_quantity(
        kinetic_energy,
        "[energy]",
        name="kinetic_energy",
        allow_negative=False,
    )
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=allow_zero_mass,
    )
    return k, m


def kinetic_energy(
    mass: QuantityLike,
    velocity: QuantityLike,
) -> pint.Quantity:
    """Compute translational kinetic energy: ``K = 0.5 * m * v^2``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        velocity: Object velocity (dimension ``[velocity]``).

    Returns:
        Kinetic energy as a ``pint.Quantity`` in ``joule``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    v = validate_quantity(velocity, "[velocity]", name="velocity")
    return to_unit(0.5 * m * (v**2), "J")


def kinetic_energy_from_momentum(
    momentum: QuantityLike,
    mass: QuantityLike,
) -> pint.Quantity:
    """Compute kinetic energy from linear momentum: ``K = p^2 / (2 * m)``.

    Args:
        momentum: Linear momentum (compatible with ``N * s`` or ``kg * m / s``).
        mass: Object mass (dimension ``[mass]``, ``> 0``).

    Returns:
        Kinetic energy as a ``pint.Quantity`` in ``joule``.
    """
    p = validate_quantity(momentum, "N * s", name="momentum")
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    return to_unit((p**2) / (2.0 * m), "J")


def speed_from_kinetic_energy(
    kinetic_energy: QuantityLike,
    mass: QuantityLike,
) -> pint.Quantity:
    """Compute speed magnitude from kinetic energy: ``v = sqrt(2 * K / m)``.

    Args:
        kinetic_energy: Kinetic energy (dimension ``[energy]``, ``>= 0``).
        mass: Object mass (dimension ``[mass]``, ``> 0``).

    Returns:
        Non-negative speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    k, m = _validate_energy_and_mass(kinetic_energy, mass, allow_zero_mass=False)
    speed_sq = float(to_si(2.0 * k / m).magnitude)
    return Quantity(math.sqrt(speed_sq), "m / s")


def momentum_from_kinetic_energy(
    kinetic_energy: QuantityLike,
    mass: QuantityLike,
) -> pint.Quantity:
    """Compute linear momentum magnitude from kinetic energy: ``p = sqrt(2 * m * K)``.

    Args:
        kinetic_energy: Kinetic energy (dimension ``[energy]``, ``>= 0``).
        mass: Object mass (dimension ``[mass]``, ``>= 0``).

    Returns:
        Non-negative momentum magnitude as an SI base ``pint.Quantity``
        (``kilogram * meter / second``).
    """
    k, m = _validate_energy_and_mass(kinetic_energy, mass, allow_zero_mass=True)
    momentum_sq = float(to_si(2.0 * m * k).magnitude)
    return Quantity(math.sqrt(momentum_sq), "kg * m / s")


__all__ = [
    "kinetic_energy",
    "kinetic_energy_from_momentum",
    "momentum_from_kinetic_energy",
    "speed_from_kinetic_energy",
]
