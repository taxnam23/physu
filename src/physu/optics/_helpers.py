"""Shared internal validation and trigonometric helpers for physu.optics."""

from __future__ import annotations

import math
from typing import Any

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity

_TRIG_ZERO_TOL = 1e-15


def _to_clean_unit(quantity: Any, target_unit: str) -> pint.Quantity:
    """Convert ``quantity`` to ``target_unit`` and normalize ``-0.0`` to ``0.0``."""
    result = to_unit(quantity, target_unit)
    return Quantity(0.0, target_unit) if result.magnitude == 0 else result


def _validate_refractive_index(
    value: QuantityLike,
    *,
    name: str = "refractive_index",
) -> float:
    """Validate a strictly positive dimensionless refractive index."""
    n = validate_quantity(
        value,
        "dimensionless",
        name=name,
        allow_negative=False,
        allow_zero=False,
    )
    return float(n.to("dimensionless").magnitude)


def _validate_integer_order(
    order: QuantityLike,
    *,
    min_value: int = 0,
    name: str = "order",
) -> int:
    """Validate a non-negative (or positive) integer diffraction/interference order."""
    q = validate_quantity(
        order,
        "dimensionless",
        name=name,
        allow_negative=False,
        allow_zero=(min_value == 0),
    )
    mag = float(q.to("dimensionless").magnitude)
    if not mag.is_integer() or int(mag) < min_value:
        raise PhysicalDomainError(
            f"Parameter '{name}' must be an integer >= {min_value}, got {order}."
        )
    return int(mag)


def _arcsin_ratio(ratio: float, *, error_message: str) -> pint.Quantity:
    """Compute ``arcsin(ratio)`` in ``radian`` after checking ``|ratio| <= 1``."""
    if abs(ratio) > 1.0:
        if math.isclose(abs(ratio), 1.0, abs_tol=1e-12):
            ratio = math.copysign(1.0, ratio)
        else:
            raise PhysicalDomainError(error_message)
    angle = math.asin(ratio)
    return Quantity(0.0 if abs(angle) < _TRIG_ZERO_TOL else angle, "rad")


def _cos_squared(angle: QuantityLike, *, name: str = "angle") -> float:
    """Validate ``angle`` against ``rad`` and compute ``cos(angle)^2`` with zero snapping."""
    rad = float(validate_quantity(angle, "rad", name=name).to("rad").magnitude)
    cos_val = math.cos(rad)
    if abs(cos_val) < _TRIG_ZERO_TOL:
        return 0.0
    return cos_val * cos_val
