"""Shared internal validation and conversion helpers for physu.fluids."""

from __future__ import annotations

import math
from typing import Any

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity

_TRIG_ZERO_TOL = 1e-15
_ANGLE_TOL = 1e-12


def _to_clean_unit(quantity: Any, target_unit: str) -> pint.Quantity:
    """Convert ``quantity`` to ``target_unit`` and normalize ``-0.0`` to ``0.0``."""
    result = to_unit(quantity, target_unit)
    return Quantity(0.0, target_unit) if result.magnitude == 0 else result


def _validate_density(
    density: QuantityLike,
    *,
    name: str = "density",
    allow_zero: bool = True,
) -> pint.Quantity:
    """Validate non-negative or strictly positive mass density (``[density]``)."""
    return validate_quantity(
        density,
        "[density]",
        name=name,
        allow_negative=False,
        allow_zero=allow_zero,
    )


def _validate_positive_gravity(gravity: QuantityLike) -> pint.Quantity:
    """Validate strictly positive gravitational acceleration."""
    return validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )


def _validate_viscosity(
    viscosity: QuantityLike,
    *,
    name: str = "dynamic_viscosity",
    allow_zero: bool = True,
) -> pint.Quantity:
    """Validate non-negative or strictly positive dynamic viscosity (``[viscosity]``)."""
    return validate_quantity(
        viscosity,
        "[viscosity]",
        name=name,
        allow_negative=False,
        allow_zero=allow_zero,
    )


def _cos_contact_angle(contact_angle: QuantityLike) -> float:
    """Validate contact angle in ``[0, 180 deg]`` and return ``cos(contact_angle)``."""
    angle_rad = float(
        validate_quantity(
            contact_angle,
            "rad",
            name="contact_angle",
            allow_negative=False,
        )
        .to("rad")
        .magnitude
    )
    if angle_rad > math.pi + _ANGLE_TOL:
        raise PhysicalDomainError(
            "Parameter 'contact_angle' must lie within [0 deg, 180 deg]."
        )
    angle_rad = min(angle_rad, math.pi)
    cos_val = math.cos(angle_rad)
    return 0.0 if abs(cos_val) < _TRIG_ZERO_TOL else cos_val
