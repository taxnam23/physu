"""Shared internal validation helpers for physu.modern."""

from __future__ import annotations

import math
from typing import Any

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity


def _to_clean_unit(quantity: Any, target_unit: str) -> pint.Quantity:
    """Convert ``quantity`` to ``target_unit`` and normalize ``-0.0`` to ``0.0``."""
    result = to_unit(quantity, target_unit)
    return Quantity(0.0, target_unit) if result.magnitude == 0 else result


def _validate_positive_c(speed_of_light: QuantityLike) -> pint.Quantity:
    """Validate strictly positive speed of light (``[velocity]``, ``> 0``)."""
    return validate_quantity(
        speed_of_light,
        "[velocity]",
        name="speed_of_light",
        allow_negative=False,
        allow_zero=False,
    )


def _subluminal_beta(
    speed: QuantityLike,
    speed_of_light: QuantityLike,
    *,
    speed_name: str = "speed",
    allow_negative: bool = True,
) -> float:
    """Validate speed ``|v| < c`` and return dimensionless ``beta = v / c``."""
    v = validate_quantity(
        speed,
        "[velocity]",
        name=speed_name,
        allow_negative=allow_negative,
    )
    c = _validate_positive_c(speed_of_light)
    beta = float(to_si(v / c).magnitude)
    if abs(beta) >= 1.0:
        raise PhysicalDomainError(
            f"Parameter '{speed_name}' magnitude must be strictly less than speed_of_light."
        )
    return beta


def _validate_positive_int(
    value: QuantityLike,
    *,
    name: str,
) -> int:
    """Validate a dimensionless integer ``>= 1`` (e.g., quantum or atomic number)."""
    q = validate_quantity(
        value,
        "dimensionless",
        name=name,
        allow_negative=False,
        allow_zero=False,
    )
    mag = float(q.to("dimensionless").magnitude)
    if not mag.is_integer():
        raise PhysicalDomainError(
            f"Parameter '{name}' must be a positive integer, got {value}."
        )
    return int(mag)


def _resolve_decay_constant(
    half_life: QuantityLike | None,
    decay_constant_value: QuantityLike | None,
) -> pint.Quantity:
    """Resolve decay constant ``lambda`` in ``1 / second`` from half-life or rate."""
    if (half_life is None) == (decay_constant_value is None):
        raise ValueError("Provide exactly one of 'half_life' or 'decay_constant'.")
    if half_life is not None:
        t_half = validate_quantity(
            half_life,
            "[time]",
            name="half_life",
            allow_negative=False,
            allow_zero=False,
        )
        return _to_clean_unit(math.log(2.0) / t_half, "1 / s")
    assert decay_constant_value is not None
    lam = validate_quantity(
        decay_constant_value,
        "1 / s",
        name="decay_constant",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(lam, "1 / s")
