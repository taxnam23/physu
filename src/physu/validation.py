"""Dimensional and physical domain validation helpers for physu."""

from __future__ import annotations

import math

import pint

from .exceptions import PhysicalDomainError
from .units import QuantityLike, check_dimension, to_quantity


def _ensure_finite_scalar(quantity: pint.Quantity, *, name: str, allow_infinite: bool) -> None:
    """Validate that a scalar quantity magnitude is not NaN (and finite when required)."""
    magnitude = quantity.magnitude
    if isinstance(magnitude, (int, float)):
        if math.isnan(magnitude):
            raise PhysicalDomainError(f"Parameter '{name}' cannot be NaN.")
        if not allow_infinite and math.isinf(magnitude):
            raise PhysicalDomainError(
                f"Parameter '{name}' must be finite, got {quantity}."
            )


def _check_domain(
    quantity: pint.Quantity,
    *,
    name: str,
    allow_negative: bool,
    allow_zero: bool,
    allow_infinite: bool,
) -> pint.Quantity:
    """Check finiteness, sign, and zero constraints on a normalized Quantity."""
    _ensure_finite_scalar(quantity, name=name, allow_infinite=allow_infinite)
    magnitude = quantity.magnitude
    if not allow_negative and magnitude < 0:
        raise PhysicalDomainError(
            f"Parameter '{name}' cannot be negative, got {quantity}."
        )
    if not allow_zero and magnitude == 0:
        raise PhysicalDomainError(
            f"Parameter '{name}' cannot be zero, got {quantity}."
        )
    return quantity


def validate_quantity(
    value: QuantityLike,
    expected_dimension: str | pint.Unit,
    *,
    name: str = "value",
    allow_negative: bool = True,
    allow_zero: bool = True,
    allow_infinite: bool = False,
) -> pint.Quantity:
    """Normalize ``value``, check its dimension, and enforce physical domain rules.

    Args:
        value: Input quantity-like value.
        expected_dimension: Expected Pint dimensionality (e.g. ``"[length]"``)
            or reference ``pint.Unit``.
        name: Parameter name included in error messages.
        allow_negative: Whether negative magnitudes are physically valid.
        allow_zero: Whether a zero magnitude is physically valid.
        allow_infinite: Whether infinite magnitudes are permitted.

    Returns:
        A validated ``pint.Quantity`` bound to ``physu.ureg``.

    Raises:
        pint.DimensionalityError: If ``value`` has incompatible dimensions.
        PhysicalDomainError: If ``value`` violates finiteness, sign, or zero rules.
    """
    quantity = check_dimension(value, expected_dimension, name=name)
    return _check_domain(
        quantity,
        name=name,
        allow_negative=allow_negative,
        allow_zero=allow_zero,
        allow_infinite=allow_infinite,
    )


def require_non_negative(
    value: QuantityLike,
    *,
    name: str = "value",
    allow_infinite: bool = False,
) -> pint.Quantity:
    """Ensure ``value`` is finite and non-negative (``>= 0``)."""
    quantity = to_quantity(value)
    return _check_domain(
        quantity,
        name=name,
        allow_negative=False,
        allow_zero=True,
        allow_infinite=allow_infinite,
    )


def require_positive(
    value: QuantityLike,
    *,
    name: str = "value",
    allow_infinite: bool = False,
) -> pint.Quantity:
    """Ensure ``value`` is finite and strictly positive (``> 0``)."""
    quantity = to_quantity(value)
    return _check_domain(
        quantity,
        name=name,
        allow_negative=False,
        allow_zero=False,
        allow_infinite=allow_infinite,
    )


def require_non_zero(
    value: QuantityLike,
    *,
    name: str = "value",
    allow_infinite: bool = False,
) -> pint.Quantity:
    """Ensure ``value`` is finite and non-zero (``!= 0``)."""
    quantity = to_quantity(value)
    return _check_domain(
        quantity,
        name=name,
        allow_negative=True,
        allow_zero=False,
        allow_infinite=allow_infinite,
    )


def require_range(
    value: QuantityLike,
    *,
    min_value: QuantityLike | None = None,
    max_value: QuantityLike | None = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    name: str = "value",
) -> pint.Quantity:
    """Ensure ``value`` is finite and lies within ``[min_value, max_value]``.

    Args:
        value: Input quantity-like value.
        min_value: Optional lower bound (must be dimensionally compatible).
        max_value: Optional upper bound (must be dimensionally compatible).
        inclusive_min: Whether ``value == min_value`` is allowed.
        inclusive_max: Whether ``value == max_value`` is allowed.
        name: Parameter name included in error messages.

    Returns:
        The normalized ``pint.Quantity``.

    Raises:
        pint.DimensionalityError: If bounds are dimensionally incompatible with ``value``.
        PhysicalDomainError: If ``value`` is NaN, infinite, or outside the bounds.
    """
    quantity = to_quantity(value)
    _ensure_finite_scalar(quantity, name=name, allow_infinite=False)

    if min_value is not None:
        lower = to_quantity(min_value)
        _ensure_finite_scalar(lower, name="min_value", allow_infinite=False)
        below = quantity < lower if inclusive_min else quantity <= lower
        if below:
            op = ">=" if inclusive_min else ">"
            raise PhysicalDomainError(
                f"Parameter '{name}' must be {op} {lower}, got {quantity}."
            )

    if max_value is not None:
        upper = to_quantity(max_value)
        _ensure_finite_scalar(upper, name="max_value", allow_infinite=False)
        above = quantity > upper if inclusive_max else quantity >= upper
        if above:
            op = "<=" if inclusive_max else "<"
            raise PhysicalDomainError(
                f"Parameter '{name}' must be {op} {upper}, got {quantity}."
            )

    return quantity


__all__ = [
    "require_non_negative",
    "require_non_zero",
    "require_positive",
    "require_range",
    "validate_quantity",
]
