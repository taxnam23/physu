"""Universal gravitation and circular dynamics formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..constants import GRAVITATIONAL_CONSTANT
from ..kinematics.circular import centripetal_acceleration
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from . import newton


def _validate_gravitational_inputs(
    mass: QuantityLike,
    distance: QuantityLike,
    gravitational_constant: QuantityLike,
    *,
    mass_name: str,
    distance_name: str,
) -> tuple[pint.Quantity, pint.Quantity, pint.Quantity]:
    """Validate non-negative mass, positive distance, and positive G."""
    m = validate_quantity(mass, "[mass]", name=mass_name, allow_negative=False)
    r = validate_quantity(
        distance,
        "[length]",
        name=distance_name,
        allow_negative=False,
        allow_zero=False,
    )
    g_const = validate_quantity(
        gravitational_constant,
        "m^3 / (kg * s^2)",
        name="gravitational_constant",
        allow_negative=False,
        allow_zero=False,
    )
    return m, r, g_const


def gravitational_field_strength(
    mass: QuantityLike,
    distance: QuantityLike,
    *,
    gravitational_constant: QuantityLike = GRAVITATIONAL_CONSTANT,
) -> pint.Quantity:
    """Compute gravitational field strength magnitude: ``g = G * M / r^2``.

    Args:
        mass: Source body mass (dimension ``[mass]``, ``>= 0``).
        distance: Distance from body center (dimension ``[length]``, ``> 0``).
        gravitational_constant: Universal gravitational constant (compatible with
            ``m^3 / (kg * s^2)``, ``> 0``, defaults to ``GRAVITATIONAL_CONSTANT``).

    Returns:
        Gravitational field strength as an SI base ``pint.Quantity``
        (``meter / second ** 2``).
    """
    m, r, g_const = _validate_gravitational_inputs(
        mass,
        distance,
        gravitational_constant,
        mass_name="mass",
        distance_name="distance",
    )
    return to_si(g_const * m / (r**2))


def gravitational_force(
    mass_1: QuantityLike,
    mass_2: QuantityLike,
    distance: QuantityLike,
    *,
    gravitational_constant: QuantityLike = GRAVITATIONAL_CONSTANT,
) -> pint.Quantity:
    """Compute Newton's universal gravitational force magnitude: ``F = G * m1 * m2 / r^2``.

    Args:
        mass_1: Mass of first body (dimension ``[mass]``, ``>= 0``).
        mass_2: Mass of second body (dimension ``[mass]``, ``>= 0``).
        distance: Center-to-center separation (dimension ``[length]``, ``> 0``).
        gravitational_constant: Universal gravitational constant (compatible with
            ``m^3 / (kg * s^2)``, ``> 0``, defaults to ``GRAVITATIONAL_CONSTANT``).

    Returns:
        Gravitational attraction magnitude as a ``pint.Quantity`` in ``newton``.
    """
    field = gravitational_field_strength(
        mass_1,
        distance,
        gravitational_constant=gravitational_constant,
    )
    return newton.force(mass_2, field)


def orbital_speed(
    central_mass: QuantityLike,
    radius: QuantityLike,
    *,
    gravitational_constant: QuantityLike = GRAVITATIONAL_CONSTANT,
) -> pint.Quantity:
    """Compute circular orbital speed: ``v = sqrt(G * M / r)``.

    Args:
        central_mass: Central body mass (dimension ``[mass]``, ``>= 0``).
        radius: Circular orbit radius (dimension ``[length]``, ``> 0``).
        gravitational_constant: Universal gravitational constant (compatible with
            ``m^3 / (kg * s^2)``, ``> 0``, defaults to ``GRAVITATIONAL_CONSTANT``).

    Returns:
        Circular orbital speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    m, r, g_const = _validate_gravitational_inputs(
        central_mass,
        radius,
        gravitational_constant,
        mass_name="central_mass",
        distance_name="radius",
    )
    speed_sq = float(to_si(g_const * m / r).magnitude)
    return Quantity(math.sqrt(speed_sq), "m / s")


def centripetal_force(
    mass: QuantityLike,
    radius: QuantityLike,
    *,
    tangential_speed: QuantityLike | None = None,
    angular_velocity: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute centripetal force magnitude: ``F_c = m * v^2 / r`` or ``F_c = m * omega^2 * r``.

    Provide exactly one of ``tangential_speed`` or ``angular_velocity``.

    Args:
        mass: Orbiting body mass (dimension ``[mass]``, ``>= 0``).
        radius: Circular path radius (dimension ``[length]``, ``> 0``).
        tangential_speed: Tangential speed (dimension ``[velocity]``).
        angular_velocity: Angular velocity (compatible with ``"rad / s"``, e.g. ``"rpm"``).

    Returns:
        Centripetal force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    a_c = centripetal_acceleration(
        radius,
        tangential_speed=tangential_speed,
        angular_velocity=angular_velocity,
    )
    return newton.force(mass, a_c)


__all__ = [
    "centripetal_force",
    "gravitational_field_strength",
    "gravitational_force",
    "orbital_speed",
]
