"""Shared internal validation and combination helpers for physu.electromagnetism."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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


def _trig_angle(
    angle: QuantityLike,
    trig_fn: Callable[[float], float],
    *,
    require_non_zero: bool = False,
) -> float:
    """Validate ``angle`` against ``rad`` and evaluate ``trig_fn`` with zero snapping."""
    angle_rad = float(validate_quantity(angle, "rad", name="angle").to("rad").magnitude)
    value = trig_fn(angle_rad)
    if abs(value) < _TRIG_ZERO_TOL:
        if require_non_zero:
            raise PhysicalDomainError(
                f"Angle produces a zero trigonometric factor ({trig_fn.__name__}(angle) == 0)."
            )
        return 0.0
    return value


def _cos_angle(angle: QuantityLike, *, require_non_zero: bool = False) -> float:
    """Validate ``angle`` and return ``cos(angle)``, snapping near-zero values."""
    return _trig_angle(angle, math.cos, require_non_zero=require_non_zero)


def _sin_angle(angle: QuantityLike, *, require_non_zero: bool = False) -> float:
    """Validate ``angle`` and return ``sin(angle)``, snapping near-zero values."""
    return _trig_angle(angle, math.sin, require_non_zero=require_non_zero)


def _validate_positive_factor(value: QuantityLike, *, name: str) -> float:
    """Validate a strictly positive dimensionless factor (e.g., relative permittivity)."""
    factor = validate_quantity(
        value,
        "dimensionless",
        name=name,
        allow_negative=False,
        allow_zero=False,
    )
    return float(factor.to("dimensionless").magnitude)


def _validate_turns(turns: QuantityLike, *, name: str = "turns") -> float:
    """Validate a strictly positive dimensionless coil turn count."""
    return _validate_positive_factor(turns, name=name)


def _validate_quantity_sequence(
    values: Sequence[QuantityLike],
    expected_dimension: str,
    *,
    item_label: str,
    allow_zero: bool,
) -> list[pint.Quantity]:
    """Validate a non-empty sequence of non-negative (or positive) quantities."""
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of {item_label}, got {type(values).__name__}."
        )
    if len(values) == 0:
        raise PhysicalDomainError(f"At least one {item_label} item must be provided.")
    return [
        validate_quantity(
            item,
            expected_dimension,
            name=f"{item_label}[{idx}]",
            allow_negative=False,
            allow_zero=allow_zero,
        )
        for idx, item in enumerate(values)
    ]


def _combine_additive(
    values: Sequence[QuantityLike],
    expected_dimension: str,
    target_unit: str,
    *,
    item_label: str,
    allow_zero: bool = True,
) -> pint.Quantity:
    """Sum a validated sequence of quantities into ``target_unit``."""
    validated = _validate_quantity_sequence(
        values,
        expected_dimension,
        item_label=item_label,
        allow_zero=allow_zero,
    )
    total = math.fsum(float(item.to(target_unit).magnitude) for item in validated)
    return Quantity(total, target_unit)


def _combine_reciprocal(
    values: Sequence[QuantityLike],
    expected_dimension: str,
    target_unit: str,
    *,
    item_label: str,
) -> pint.Quantity:
    """Combine a validated sequence of positive quantities via ``1 / sum(1 / x_i)``."""
    validated = _validate_quantity_sequence(
        values,
        expected_dimension,
        item_label=item_label,
        allow_zero=False,
    )
    reciprocal_sum = math.fsum(
        1.0 / float(item.to(target_unit).magnitude) for item in validated
    )
    return Quantity(1.0 / reciprocal_sum, target_unit)
