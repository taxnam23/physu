"""Centralized Pint unit registry and quantity conversion utilities for physu."""

from __future__ import annotations

from typing import TypeAlias

import pint

ureg: pint.UnitRegistry = pint.UnitRegistry()
Quantity = ureg.Quantity

QuantityLike: TypeAlias = pint.Quantity | str | float | int


def to_quantity(value: QuantityLike) -> pint.Quantity:
    """Normalize a quantity-like input into a ``Quantity`` bound to ``physu.ureg``.

    Args:
        value: A ``pint.Quantity`` (from ``physu.ureg`` or another registry),
            a string expression such as ``"10 m/s"``, or a real scalar
            (converted to dimensionless).

    Returns:
        A ``pint.Quantity`` belonging to ``physu.units.ureg``.

    Raises:
        TypeError: If ``value`` is a ``bool`` or an unsupported type.
        ValueError: If ``value`` is an empty or whitespace-only string.
    """
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid physical quantities.")

    if isinstance(value, Quantity):
        return value

    if isinstance(value, pint.Quantity):
        return Quantity(value.magnitude, str(value.units))

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Quantity string cannot be empty.")
        return Quantity(stripped)

    if isinstance(value, (int, float)):
        return Quantity(value, ureg.dimensionless)

    raise TypeError(
        f"Expected a pint.Quantity, str, int, or float, got {type(value).__name__}."
    )


def check_dimension(
    value: QuantityLike,
    expected_dimension: str | pint.Unit,
    *,
    name: str = "value",
) -> pint.Quantity:
    """Validate that ``value`` is dimensionally compatible with ``expected_dimension``.

    Args:
        value: Input quantity-like value to normalize and validate.
        expected_dimension: Pint dimensionality string (e.g. ``"[length]"``)
            or reference ``pint.Unit``.
        name: Parameter name included in error messages.

    Returns:
        The normalized ``pint.Quantity`` bound to ``physu.ureg``.

    Raises:
        pint.DimensionalityError: If ``value`` does not match ``expected_dimension``.
    """
    quantity = to_quantity(value)
    expected = (
        str(expected_dimension)
        if isinstance(expected_dimension, pint.Unit)
        else expected_dimension
    )
    if not quantity.is_compatible_with(expected):
        raise pint.DimensionalityError(
            quantity.units,
            expected,
            dim1=quantity.dimensionality,
            dim2=ureg.get_dimensionality(expected),
            extra_msg=f" (for parameter '{name}')",
        )
    return quantity


def convert_to(
    value: QuantityLike,
    target_unit: str | pint.Unit,
    *,
    name: str = "value",
) -> pint.Quantity:
    """Normalize ``value`` and convert it to ``target_unit``.

    Args:
        value: Input quantity-like value.
        target_unit: Target unit string (e.g. ``"m/s"``) or ``pint.Unit``.
        name: Parameter name included in error messages on mismatch.

    Returns:
        A ``pint.Quantity`` converted to ``target_unit``.

    Raises:
        pint.DimensionalityError: If ``value`` is incompatible with ``target_unit``.
    """
    target = str(target_unit)
    quantity = check_dimension(value, ureg.Unit(target), name=name)
    return quantity.to(target)


__all__ = [
    "Quantity",
    "QuantityLike",
    "check_dimension",
    "convert_to",
    "to_quantity",
    "ureg",
]
