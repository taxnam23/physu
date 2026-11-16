"""Hooke's law and spring combination formulas with Pint unit validation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity


def _validate_spring_constants(
    spring_constants: Sequence[QuantityLike],
    *,
    allow_zero: bool,
) -> list[pint.Quantity]:
    """Validate a non-empty sequence of spring constants compatible with ``N / m``."""
    if isinstance(spring_constants, (str, bytes)):
        raise TypeError(
            "Expected a sequence of spring constants, got "
            f"{type(spring_constants).__name__}."
        )
    if len(spring_constants) == 0:
        raise PhysicalDomainError("At least one spring constant must be provided.")
    return [
        validate_quantity(
            k,
            "N / m",
            name=f"spring_constants[{idx}]",
            allow_negative=False,
            allow_zero=allow_zero,
        )
        for idx, k in enumerate(spring_constants)
    ]


def spring_force(
    spring_constant: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute Hooke's law restoring force: ``F = -k * x``.

    Args:
        spring_constant: Spring stiffness (compatible with ``N / m``, ``>= 0``).
        displacement: Spring displacement from equilibrium (``[length]``).

    Returns:
        Signed restoring force as a ``pint.Quantity`` in ``newton``.
    """
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
    )
    x = validate_quantity(displacement, "[length]", name="displacement")
    result = to_unit(-k * x, "N")
    return Quantity(0.0, "N") if result.magnitude == 0 else result


def spring_force_magnitude(
    spring_constant: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute elastic force magnitude: ``|F| = k * |delta_x|``.

    Args:
        spring_constant: Spring stiffness (compatible with ``N / m``, ``>= 0``).
        displacement: Spring deformation or displacement (``[length]``).

    Returns:
        Force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
    )
    x = validate_quantity(displacement, "[length]", name="displacement")
    return to_unit(k * abs(x), "N")


def spring_constant(
    force: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute spring constant from Hooke's law: ``k = |F| / |delta_x|``.

    Args:
        force: Applied or restoring elastic force (``[force]``).
        displacement: Non-zero spring deformation (``[length]``, ``!= 0``).

    Returns:
        Spring constant as a ``pint.Quantity`` in ``newton / meter``.
    """
    f = validate_quantity(force, "[force]", name="force")
    x = validate_quantity(
        displacement,
        "[length]",
        name="displacement",
        allow_zero=False,
    )
    return to_unit(abs(f / x), "N / m")


def spring_displacement(
    force: QuantityLike,
    spring_constant: QuantityLike,
) -> pint.Quantity:
    """Compute spring displacement from Hooke's law restoring force: ``x = -F / k``.

    Args:
        force: Restoring force exerted by the spring (``[force]``).
        spring_constant: Positive spring stiffness (compatible with ``N / m``, ``> 0``).

    Returns:
        Signed displacement from equilibrium as a ``pint.Quantity`` in ``meter``.
    """
    f = validate_quantity(force, "[force]", name="force")
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
        allow_zero=False,
    )
    result = to_si(-f / k)
    return Quantity(0.0, "m") if result.magnitude == 0 else result


def series_spring_constant(spring_constants: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute effective spring constant for springs in series: ``1 / k_eq = sum(1 / k_i)``.

    Args:
        spring_constants: Non-empty sequence of positive spring constants
            (compatible with ``N / m``, each ``> 0``).

    Returns:
        Equivalent spring constant as a ``pint.Quantity`` in ``newton / meter``.
    """
    validated = _validate_spring_constants(spring_constants, allow_zero=False)
    reciprocal_sum = math.fsum(
        1.0 / float(k.to("N / m").magnitude) for k in validated
    )
    return Quantity(1.0 / reciprocal_sum, "N / m")


def parallel_spring_constant(spring_constants: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute effective spring constant for springs in parallel: ``k_eq = sum(k_i)``.

    Args:
        spring_constants: Non-empty sequence of non-negative spring constants
            (compatible with ``N / m``, each ``>= 0``).

    Returns:
        Equivalent spring constant as a ``pint.Quantity`` in ``newton / meter``.
    """
    validated = _validate_spring_constants(spring_constants, allow_zero=True)
    total = math.fsum(float(k.to("N / m").magnitude) for k in validated)
    return Quantity(total, "N / m")


__all__ = [
    "parallel_spring_constant",
    "series_spring_constant",
    "spring_constant",
    "spring_displacement",
    "spring_force",
    "spring_force_magnitude",
]
