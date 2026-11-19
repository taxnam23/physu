"""Thin lens, spherical mirror, magnification, and optical power formulas."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_refractive_index

_RECIPROCAL_TOL = 1e-15


def _reciprocal_combine_meters(
    term_1: QuantityLike,
    term_2: QuantityLike,
    *,
    name_1: str,
    name_2: str,
    sign: float,
    error_message: str,
) -> pint.Quantity:
    """Compute ``1 / (1 / term_1 + sign * (1 / term_2))`` in ``meter``."""
    q1_m = float(
        validate_quantity(term_1, "[length]", name=name_1, allow_zero=False)
        .to("m")
        .magnitude
    )
    q2_m = float(
        validate_quantity(term_2, "[length]", name=name_2, allow_zero=False)
        .to("m")
        .magnitude
    )
    inv_sum = (1.0 / q1_m) + sign * (1.0 / q2_m)
    if abs(inv_sum) < _RECIPROCAL_TOL:
        raise PhysicalDomainError(error_message)
    return Quantity(1.0 / inv_sum, "m")


def thin_lens_image_distance(
    focal_length: QuantityLike,
    object_distance: QuantityLike,
) -> pint.Quantity:
    """Compute image distance via the thin-lens/mirror equation: ``1 / d_i = 1 / f - 1 / d_o``.

    Args:
        focal_length: Signed focal length (dimension ``[length]``, ``!= 0``).
        object_distance: Signed object distance (dimension ``[length]``, ``!= 0``).

    Returns:
        Image distance as a ``pint.Quantity`` in ``meter``.

    Raises:
        PhysicalDomainError: If ``object_distance == focal_length`` (image at infinity)
            or either input is zero.
    """
    return _reciprocal_combine_meters(
        focal_length,
        object_distance,
        name_1="focal_length",
        name_2="object_distance",
        sign=-1.0,
        error_message=(
            "Image forms at infinity when object_distance equals focal_length "
            "(1 / f - 1 / d_o == 0)."
        ),
    )


def thin_lens_focal_length(
    object_distance: QuantityLike,
    image_distance: QuantityLike,
) -> pint.Quantity:
    """Compute focal length via the thin-lens/mirror equation: ``1 / f = 1 / d_o + 1 / d_i``.

    Args:
        object_distance: Signed object distance (dimension ``[length]``, ``!= 0``).
        image_distance: Signed image distance (dimension ``[length]``, ``!= 0``).

    Returns:
        Focal length as a ``pint.Quantity`` in ``meter``.

    Raises:
        PhysicalDomainError: If ``1 / object_distance + 1 / image_distance == 0``
            or either input is zero.
    """
    return _reciprocal_combine_meters(
        object_distance,
        image_distance,
        name_1="object_distance",
        name_2="image_distance",
        sign=1.0,
        error_message=(
            "Focal length is undefined for an afocal configuration "
            "(1 / d_o + 1 / d_i == 0)."
        ),
    )


def lateral_magnification(
    object_distance: QuantityLike,
    image_distance: QuantityLike,
) -> pint.Quantity:
    """Compute lateral transverse magnification: ``M = -d_i / d_o``.

    Args:
        object_distance: Non-zero object distance (dimension ``[length]``, ``!= 0``).
        image_distance: Image distance (dimension ``[length]``).

    Returns:
        Dimensionless lateral magnification as a ``pint.Quantity``
        (negative indicates an inverted image).
    """
    d_o = validate_quantity(
        object_distance,
        "[length]",
        name="object_distance",
        allow_zero=False,
    )
    d_i = validate_quantity(image_distance, "[length]", name="image_distance")
    mag = to_si(-d_i / d_o)
    return Quantity(0.0, "dimensionless") if mag.magnitude == 0 else mag


def optical_power(focal_length: QuantityLike) -> pint.Quantity:
    """Compute optical power (diopters in SI inverse meters): ``P = 1 / f``.

    Args:
        focal_length: Non-zero focal length (dimension ``[length]``, ``!= 0``).

    Returns:
        Optical power as a ``pint.Quantity`` in ``1 / meter``.
    """
    f = validate_quantity(focal_length, "[length]", name="focal_length", allow_zero=False)
    return _to_clean_unit(1.0 / f, "1 / m")


def combined_optical_power(focal_lengths: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute combined optical power of thin lenses in contact: ``P_eq = sum(1 / f_i)``.

    Args:
        focal_lengths: Non-empty sequence of non-zero focal lengths
            (dimension ``[length]``).

    Returns:
        Combined optical power as a ``pint.Quantity`` in ``1 / meter``.
    """
    if isinstance(focal_lengths, (str, bytes)):
        raise TypeError(
            f"Expected a sequence of focal lengths, got {type(focal_lengths).__name__}."
        )
    if len(focal_lengths) == 0:
        raise PhysicalDomainError("At least one focal length must be provided.")
    total = math.fsum(
        float(optical_power(f_i).magnitude) for f_i in focal_lengths
    )
    return Quantity(0.0 if total == 0 else total, "1 / m")


def lensmaker_focal_length(
    refractive_index: QuantityLike,
    radius_1: QuantityLike,
    radius_2: QuantityLike,
    *,
    surrounding_index: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute focal length via the thin lensmaker's equation.

    Uses ``1 / f = (n / n_0 - 1) * (1 / R_1 - 1 / R_2)`` under the Cartesian
    sign convention (e.g., ``R_1 > 0`` and ``R_2 < 0`` for a biconvex lens).

    Args:
        refractive_index: Lens material refractive index (dimensionless, ``> 0``).
        radius_1: Front surface radius of curvature (dimension ``[length]``, ``!= 0``).
        radius_2: Back surface radius of curvature (dimension ``[length]``, ``!= 0``).
        surrounding_index: Surrounding medium refractive index (dimensionless,
            ``> 0``, defaults to ``1.0``).

    Returns:
        Focal length as a ``pint.Quantity`` in ``meter``.

    Raises:
        PhysicalDomainError: If ``refractive_index == surrounding_index`` or
            ``1 / R_1 - 1 / R_2 == 0``.
    """
    n = _validate_refractive_index(refractive_index, name="refractive_index")
    n0 = _validate_refractive_index(surrounding_index, name="surrounding_index")
    r1_m = float(
        validate_quantity(radius_1, "[length]", name="radius_1", allow_zero=False)
        .to("m")
        .magnitude
    )
    r2_m = float(
        validate_quantity(radius_2, "[length]", name="radius_2", allow_zero=False)
        .to("m")
        .magnitude
    )
    delta_n = (n / n0) - 1.0
    if abs(delta_n) < _RECIPROCAL_TOL:
        raise PhysicalDomainError(
            "Lens refractive index cannot equal surrounding index (zero optical power)."
        )
    curvature_diff = (1.0 / r1_m) - (1.0 / r2_m)
    if abs(curvature_diff) < _RECIPROCAL_TOL:
        raise PhysicalDomainError(
            "Surface curvature difference (1 / radius_1 - 1 / radius_2) cannot be zero."
        )
    return Quantity(1.0 / (delta_n * curvature_diff), "m")


def mirror_focal_length(radius_of_curvature: QuantityLike) -> pint.Quantity:
    """Compute spherical mirror focal length: ``f = R / 2``.

    Args:
        radius_of_curvature: Signed mirror radius of curvature (dimension ``[length]``).

    Returns:
        Mirror focal length as a ``pint.Quantity`` in ``meter``.
    """
    r = validate_quantity(radius_of_curvature, "[length]", name="radius_of_curvature")
    return _to_clean_unit(0.5 * r, "m")


__all__ = [
    "combined_optical_power",
    "lateral_magnification",
    "lensmaker_focal_length",
    "mirror_focal_length",
    "optical_power",
    "thin_lens_focal_length",
    "thin_lens_image_distance",
]
