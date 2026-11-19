"""Capillarity formulas: surface tension force, Young-Laplace pressure, and capillary rise."""

from __future__ import annotations

import pint

from ..constants import STANDARD_GRAVITY
from ..exceptions import PhysicalDomainError
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import (
    _cos_contact_angle,
    _to_clean_unit,
    _validate_density,
    _validate_positive_gravity,
)


def _validate_surface_tension(surface_tension: QuantityLike) -> pint.Quantity:
    """Validate non-negative surface tension compatible with ``N / m``."""
    return validate_quantity(
        surface_tension,
        "N / m",
        name="surface_tension",
        allow_negative=False,
    )


def surface_tension_force(
    surface_tension: QuantityLike,
    contact_length: QuantityLike,
) -> pint.Quantity:
    """Compute surface tension force along a contact line: ``F = gamma * L``.

    Args:
        surface_tension: Fluid surface tension (compatible with ``N / m``, ``>= 0``).
        contact_length: Total wetting perimeter or wire contact length
            (dimension ``[length]``, ``>= 0``).

    Returns:
        Surface tension force as a ``pint.Quantity`` in ``newton``.
    """
    gamma = _validate_surface_tension(surface_tension)
    length_q = validate_quantity(
        contact_length,
        "[length]",
        name="contact_length",
        allow_negative=False,
    )
    return _to_clean_unit(gamma * length_q, "N")


def laplace_bubble_pressure(
    surface_tension: QuantityLike,
    radius: QuantityLike,
    *,
    interfaces: QuantityLike = 2,
) -> pint.Quantity:
    """Compute Young-Laplace excess internal pressure: ``delta_P = 2 * interfaces * gamma / R``.

    Use ``interfaces=1`` for a liquid droplet in air or submerged gas bubble
    (``2 * gamma / R``), and ``interfaces=2`` for a soap bubble in air
    (``4 * gamma / R``).

    Args:
        surface_tension: Fluid surface tension (compatible with ``N / m``, ``>= 0``).
        radius: Spherical droplet or bubble radius (dimension ``[length]``, ``> 0``).
        interfaces: Number of spherical gas-liquid interfaces (``1`` or ``2``,
            defaults to ``2``).

    Returns:
        Excess internal pressure as a ``pint.Quantity`` in ``pascal``.

    Raises:
        PhysicalDomainError: If ``interfaces`` is not ``1`` or ``2``, or inputs
            violate physical bounds.
    """
    gamma = _validate_surface_tension(surface_tension)
    r = validate_quantity(
        radius,
        "[length]",
        name="radius",
        allow_negative=False,
        allow_zero=False,
    )
    if_q = validate_quantity(
        interfaces,
        "dimensionless",
        name="interfaces",
        allow_negative=False,
        allow_zero=False,
    )
    if_mag = float(if_q.to("dimensionless").magnitude)
    if not if_mag.is_integer() or int(if_mag) not in (1, 2):
        raise PhysicalDomainError(
            f"Parameter 'interfaces' must be 1 or 2, got {interfaces}."
        )
    return _to_clean_unit((2.0 * int(if_mag) * gamma) / r, "Pa")


def capillary_rise_height(
    surface_tension: QuantityLike,
    fluid_density: QuantityLike,
    tube_radius: QuantityLike,
    *,
    contact_angle: QuantityLike = "0 deg",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute capillary rise or depression height: ``h = 2 * gamma * cos(theta) / (rho * g * r)``.

    Wetting liquids (``contact_angle < 90 deg``) yield positive rise; non-wetting
    liquids (``contact_angle > 90 deg``, e.g. mercury on glass) yield negative depression.

    Args:
        surface_tension: Fluid surface tension (compatible with ``N / m``, ``>= 0``).
        fluid_density: Fluid mass density (dimension ``[density]``, ``> 0``).
        tube_radius: Capillary tube inner radius (dimension ``[length]``, ``> 0``).
        contact_angle: Liquid-wall contact angle in ``[0 deg, 180 deg]``
            (compatible with ``"rad"``, defaults to ``"0 deg"``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Signed capillary rise height as a ``pint.Quantity`` in ``meter``.
    """
    gamma = _validate_surface_tension(surface_tension)
    rho = _validate_density(fluid_density, name="fluid_density", allow_zero=False)
    r = validate_quantity(
        tube_radius,
        "[length]",
        name="tube_radius",
        allow_negative=False,
        allow_zero=False,
    )
    cos_theta = _cos_contact_angle(contact_angle)
    g = _validate_positive_gravity(gravity)
    return _to_clean_unit((2.0 * gamma * cos_theta) / (rho * g * r), "m")


__all__ = [
    "capillary_rise_height",
    "laplace_bubble_pressure",
    "surface_tension_force",
]
