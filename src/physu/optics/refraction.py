"""Optical media, Snell's law refraction, critical angle, and Brewster angle formulas."""

from __future__ import annotations

import math

import pint

from ..constants import SPEED_OF_LIGHT
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _arcsin_ratio, _to_clean_unit, _validate_refractive_index

_ANGLE_TOL = 1e-12


def _validate_positive_c(speed_of_light: QuantityLike) -> pint.Quantity:
    """Validate strictly positive speed of light."""
    return validate_quantity(
        speed_of_light,
        "[velocity]",
        name="speed_of_light",
        allow_negative=False,
        allow_zero=False,
    )


def _validate_interface_indices(
    n_1: QuantityLike,
    n_2: QuantityLike,
) -> tuple[float, float]:
    """Validate positive refractive indices for an optical interface."""
    return (
        _validate_refractive_index(n_1, name="n_1"),
        _validate_refractive_index(n_2, name="n_2"),
    )


def speed_in_medium(
    refractive_index: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute phase speed of light in an optical medium: ``v = c / n``.

    Args:
        refractive_index: Medium refractive index (dimensionless, ``> 0``).
        speed_of_light: Speed of light in vacuum (dimension ``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Speed in the medium as a ``pint.Quantity`` in ``meter / second``.
    """
    n = _validate_refractive_index(refractive_index, name="refractive_index")
    c = _validate_positive_c(speed_of_light)
    return _to_clean_unit(c / n, "m / s")


def wavelength_in_medium(
    vacuum_wavelength: QuantityLike,
    refractive_index: QuantityLike,
) -> pint.Quantity:
    """Compute light wavelength inside an optical medium: ``lambda_n = lambda_0 / n``.

    Args:
        vacuum_wavelength: Wavelength in vacuum (dimension ``[length]``, ``>= 0``).
        refractive_index: Medium refractive index (dimensionless, ``> 0``).

    Returns:
        Wavelength in the medium as a ``pint.Quantity`` in ``meter``.
    """
    lam_0 = validate_quantity(
        vacuum_wavelength,
        "[length]",
        name="vacuum_wavelength",
        allow_negative=False,
    )
    n = _validate_refractive_index(refractive_index, name="refractive_index")
    return _to_clean_unit(lam_0 / n, "m")


def refractive_index(
    medium_speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute refractive index from light speed in a medium: ``n = c / v``.

    Args:
        medium_speed: Phase speed of light in the medium (``[velocity]``, ``> 0``).
        speed_of_light: Speed of light in vacuum (dimension ``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Dimensionless refractive index as a ``pint.Quantity``.
    """
    v = validate_quantity(
        medium_speed,
        "[velocity]",
        name="medium_speed",
        allow_negative=False,
        allow_zero=False,
    )
    c = _validate_positive_c(speed_of_light)
    return to_si(c / v)


def snell_refraction_angle(
    incident_angle: QuantityLike,
    n_1: QuantityLike,
    n_2: QuantityLike,
) -> pint.Quantity:
    """Compute refracted angle via Snell's law: ``n_1 * sin(theta_1) = n_2 * sin(theta_2)``.

    Args:
        incident_angle: Angle of incidence from the surface normal in ``[0, 90 deg]``
            (compatible with ``"rad"``).
        n_1: Incident medium refractive index (dimensionless, ``> 0``).
        n_2: Transmitted medium refractive index (dimensionless, ``> 0``).

    Returns:
        Refracted angle as a ``pint.Quantity`` in ``radian`` (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``incident_angle`` is outside ``[0, 90 deg]`` or
            if total internal reflection occurs (``(n_1 / n_2) * sin(theta_1) > 1``).
    """
    theta_1 = float(
        validate_quantity(
            incident_angle,
            "rad",
            name="incident_angle",
            allow_negative=False,
        )
        .to("rad")
        .magnitude
    )
    if theta_1 > (0.5 * math.pi) + _ANGLE_TOL:
        raise PhysicalDomainError(
            "Parameter 'incident_angle' must lie within [0 deg, 90 deg]."
        )
    theta_1 = min(theta_1, 0.5 * math.pi)
    idx_1, idx_2 = _validate_interface_indices(n_1, n_2)
    sin_theta_2 = (idx_1 / idx_2) * math.sin(theta_1)
    return _arcsin_ratio(
        sin_theta_2,
        error_message=(
            "Total internal reflection occurs: incident angle exceeds the critical angle."
        ),
    )


def critical_angle(
    n_1: QuantityLike,
    n_2: QuantityLike,
) -> pint.Quantity:
    """Compute critical angle for total internal reflection: ``theta_c = arcsin(n_2 / n_1)``.

    Args:
        n_1: Incident (denser) medium refractive index (dimensionless, ``> 0``).
        n_2: Transmitted (less dense) medium refractive index (dimensionless, ``> 0``,
            satisfying ``n_2 <= n_1``).

    Returns:
        Critical angle as a ``pint.Quantity`` in ``radian`` (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``n_1 < n_2``.
    """
    idx_1, idx_2 = _validate_interface_indices(n_1, n_2)
    return _arcsin_ratio(
        idx_2 / idx_1,
        error_message=(
            "Critical angle is only defined when incident medium n_1 >= transmitted medium n_2."
        ),
    )


def brewster_angle(
    n_1: QuantityLike,
    n_2: QuantityLike,
) -> pint.Quantity:
    """Compute Brewster polarizing angle: ``theta_B = arctan(n_2 / n_1)``.

    Args:
        n_1: Incident medium refractive index (dimensionless, ``> 0``).
        n_2: Reflecting medium refractive index (dimensionless, ``> 0``).

    Returns:
        Brewster angle as a ``pint.Quantity`` in ``radian`` (convertible to ``degree``).
    """
    idx_1, idx_2 = _validate_interface_indices(n_1, n_2)
    return Quantity(math.atan(idx_2 / idx_1), "rad")


__all__ = [
    "brewster_angle",
    "critical_angle",
    "refractive_index",
    "snell_refraction_angle",
    "speed_in_medium",
    "wavelength_in_medium",
]
