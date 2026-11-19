"""Optical polarization formulas: Malus's law, polarizer stacks, and degree of polarization."""

from __future__ import annotations

from collections.abc import Sequence

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _cos_squared, _to_clean_unit


def _validate_intensity(
    intensity: QuantityLike,
    *,
    name: str,
    allow_zero: bool = True,
) -> pint.Quantity:
    """Validate optical irradiance/intensity compatible with ``W / m^2``."""
    return validate_quantity(
        intensity,
        "W / m^2",
        name=name,
        allow_negative=False,
        allow_zero=allow_zero,
    )


def malus_intensity(
    incident_intensity: QuantityLike,
    angle: QuantityLike,
) -> pint.Quantity:
    """Compute transmitted polarized intensity via Malus's law: ``I = I_0 * cos(theta)^2``.

    Args:
        incident_intensity: Linearly polarized incident intensity
            (compatible with ``W / m^2``, ``>= 0``).
        angle: Angle between polarization direction and analyzer transmission axis
            (compatible with ``"rad"``).

    Returns:
        Transmitted intensity as a ``pint.Quantity`` in ``watt / meter ** 2``.
    """
    i0 = _validate_intensity(incident_intensity, name="incident_intensity")
    cos_sq = _cos_squared(angle, name="angle")
    return _to_clean_unit(i0 * cos_sq, "W / m^2")


def unpolarized_polarizer_intensity(incident_intensity: QuantityLike) -> pint.Quantity:
    """Compute intensity of unpolarized light after an ideal linear polarizer: ``I = I_0 / 2``.

    Args:
        incident_intensity: Unpolarized incident intensity
            (compatible with ``W / m^2``, ``>= 0``).

    Returns:
        Transmitted intensity as a ``pint.Quantity`` in ``watt / meter ** 2``.
    """
    i0 = _validate_intensity(incident_intensity, name="incident_intensity")
    return _to_clean_unit(0.5 * i0, "W / m^2")


def polarizer_stack_intensity(
    incident_intensity: QuantityLike,
    relative_angles: Sequence[QuantityLike],
) -> pint.Quantity:
    """Compute intensity after consecutive linear analyzers: ``I = I_0 * prod(cos(theta_k)^2)``.

    Args:
        incident_intensity: Linearly polarized incident intensity
            (compatible with ``W / m^2``, ``>= 0``).
        relative_angles: Non-empty sequence of relative angles between each
            consecutive transmission axis (compatible with ``"rad"``).

    Returns:
        Transmitted intensity after the stack as a ``pint.Quantity``
        in ``watt / meter ** 2``.
    """
    if isinstance(relative_angles, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of relative angles, got {type(relative_angles).__name__}."
        )
    if len(relative_angles) == 0:
        raise PhysicalDomainError("At least one relative angle must be provided.")

    intensity = _validate_intensity(incident_intensity, name="incident_intensity")
    for idx, angle in enumerate(relative_angles):
        cos_sq = _cos_squared(angle, name=f"relative_angles[{idx}]")
        intensity = intensity * cos_sq
    return _to_clean_unit(intensity, "W / m^2")


def polarization_degree(
    max_intensity: QuantityLike,
    min_intensity: QuantityLike,
) -> pint.Quantity:
    """Compute degree of linear polarization: ``P = (I_max - I_min) / (I_max + I_min)``.

    Args:
        max_intensity: Maximum transmitted intensity as analyzer rotates
            (compatible with ``W / m^2``, ``> 0``).
        min_intensity: Minimum transmitted intensity as analyzer rotates
            (compatible with ``W / m^2``, ``>= 0``, ``<= max_intensity``).

    Returns:
        Dimensionless degree of polarization in ``[0, 1]``.

    Raises:
        PhysicalDomainError: If ``min_intensity > max_intensity`` or bounds fail.
    """
    i_max = _validate_intensity(max_intensity, name="max_intensity", allow_zero=False)
    i_min = _validate_intensity(min_intensity, name="min_intensity", allow_zero=True)
    if float(to_si(i_min).magnitude) > float(to_si(i_max).magnitude):
        raise PhysicalDomainError("Parameter 'min_intensity' cannot exceed 'max_intensity'.")
    degree = to_si((i_max - i_min) / (i_max + i_min))
    return Quantity(0.0, "dimensionless") if degree.magnitude == 0 else degree


__all__ = [
    "malus_intensity",
    "polarization_degree",
    "polarizer_stack_intensity",
    "unpolarized_polarizer_intensity",
]
