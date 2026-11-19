"""Optical diffraction formulas: single slit, gratings, Rayleigh resolution, and Bragg's law."""

from __future__ import annotations

import pint

from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _arcsin_ratio, _to_clean_unit, _validate_integer_order
from .interference import double_slit_angle


def _diffraction_arcsin_angle(
    order: QuantityLike,
    wavelength: QuantityLike,
    spacing: QuantityLike,
    *,
    min_order: int,
    spacing_name: str,
    denominator_scale: float,
    error_message: str,
) -> pint.Quantity:
    """Compute ``arcsin((order * wavelength) / (denominator_scale * spacing))`` in ``radian``."""
    m = _validate_integer_order(order, min_value=min_order, name="order")
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    d = validate_quantity(
        spacing,
        "[length]",
        name=spacing_name,
        allow_negative=False,
        allow_zero=False,
    )
    ratio = float(to_si((m * lam) / (denominator_scale * d)).magnitude)
    return _arcsin_ratio(ratio, error_message=error_message)


def single_slit_minimum_angle(
    order: QuantityLike,
    wavelength: QuantityLike,
    slit_width: QuantityLike,
) -> pint.Quantity:
    """Compute diffraction minimum angle for a single slit: ``a * sin(theta) = m * lambda``.

    Args:
        order: Positive integer minimum order ``m >= 1``.
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        slit_width: Slit width ``a`` (dimension ``[length]``, ``> 0``).

    Returns:
        Angular position of the minimum as a ``pint.Quantity`` in ``radian``
        (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``m < 1``, non-integer, or ``m * wavelength > slit_width``.
    """
    return _diffraction_arcsin_angle(
        order,
        wavelength,
        slit_width,
        min_order=1,
        spacing_name="slit_width",
        denominator_scale=1.0,
        error_message=(
            "No diffraction minimum exists for this order (m * wavelength > slit_width)."
        ),
    )


def diffraction_grating_angle(
    order: QuantityLike,
    wavelength: QuantityLike,
    slit_spacing: QuantityLike,
) -> pint.Quantity:
    """Compute principal maximum angle for a diffraction grating: ``d * sin(theta) = m * lambda``.

    Args:
        order: Non-negative integer diffraction order ``m >= 0``.
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        slit_spacing: Grating line spacing ``d`` (dimension ``[length]``, ``> 0``).

    Returns:
        Diffraction angle as a ``pint.Quantity`` in ``radian`` (convertible to ``degree``).
    """
    return double_slit_angle(order, wavelength, slit_spacing)


def grating_spacing_from_density(lines_per_length: QuantityLike) -> pint.Quantity:
    """Compute diffraction grating spacing from line density: ``d = 1 / N``.

    Args:
        lines_per_length: Grating groove density (compatible with ``1 / m`` or
            ``1 / mm``, ``> 0``).

    Returns:
        Grating line spacing as a ``pint.Quantity`` in ``meter``.
    """
    density = validate_quantity(
        lines_per_length,
        "1 / m",
        name="lines_per_length",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(1.0 / density, "m")


def rayleigh_angular_resolution(
    wavelength: QuantityLike,
    aperture_diameter: QuantityLike,
) -> pint.Quantity:
    """Compute Rayleigh diffraction limit for a circular aperture: ``theta_R = 1.22 * lambda / D``.

    Args:
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        aperture_diameter: Circular aperture diameter ``D`` (``[length]``, ``> 0``).

    Returns:
        Minimum resolvable angular separation as a ``pint.Quantity`` in ``radian``
        (convertible to ``arcsec`` or ``degree``).
    """
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    diam = validate_quantity(
        aperture_diameter,
        "[length]",
        name="aperture_diameter",
        allow_negative=False,
        allow_zero=False,
    )
    theta_rad = 1.22 * float(to_si(lam / diam).magnitude)
    return Quantity(theta_rad, "rad")


def bragg_diffraction_angle(
    order: QuantityLike,
    wavelength: QuantityLike,
    plane_spacing: QuantityLike,
) -> pint.Quantity:
    """Compute Bragg X-ray diffraction grazing angle: ``2 * d * sin(theta) = m * lambda``.

    Args:
        order: Positive integer diffraction order ``m >= 1``.
        wavelength: Radiation wavelength (dimension ``[length]``, ``>= 0``).
        plane_spacing: Crystal interplanar spacing ``d`` (dimension ``[length]``, ``> 0``).

    Returns:
        Bragg grazing angle as a ``pint.Quantity`` in ``radian``
        (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``m < 1``, non-integer, or ``m * wavelength > 2 * plane_spacing``.
    """
    return _diffraction_arcsin_angle(
        order,
        wavelength,
        plane_spacing,
        min_order=1,
        spacing_name="plane_spacing",
        denominator_scale=2.0,
        error_message=(
            "No Bragg diffraction peak exists for this order "
            "(m * wavelength > 2 * plane_spacing)."
        ),
    )


__all__ = [
    "bragg_diffraction_angle",
    "diffraction_grating_angle",
    "grating_spacing_from_density",
    "rayleigh_angular_resolution",
    "single_slit_minimum_angle",
]
