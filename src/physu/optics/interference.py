"""Optical interference formulas: Young's double slit, thin films, and Michelson."""

from __future__ import annotations

import pint

from ..exceptions import PhysicalDomainError
from ..units import QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import (
    _arcsin_ratio,
    _to_clean_unit,
    _validate_integer_order,
    _validate_refractive_index,
)


def double_slit_angle(
    order: QuantityLike,
    wavelength: QuantityLike,
    slit_separation: QuantityLike,
) -> pint.Quantity:
    """Compute constructive interference angle for a double slit: ``d * sin(theta) = m * lambda``.

    Args:
        order: Non-negative integer interference order ``m >= 0``.
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        slit_separation: Center-to-center slit spacing ``d`` (``[length]``, ``> 0``).

    Returns:
        Angular position as a ``pint.Quantity`` in ``radian`` (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``m * wavelength > slit_separation``.
    """
    m = _validate_integer_order(order, min_value=0, name="order")
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    d = validate_quantity(
        slit_separation,
        "[length]",
        name="slit_separation",
        allow_negative=False,
        allow_zero=False,
    )
    ratio = float(to_si((m * lam) / d).magnitude)
    return _arcsin_ratio(
        ratio,
        error_message=(
            "No constructive interference maximum exists for this order "
            "(m * wavelength > slit_separation)."
        ),
    )


def double_slit_fringe_position(
    order: QuantityLike,
    wavelength: QuantityLike,
    screen_distance: QuantityLike,
    slit_separation: QuantityLike,
) -> pint.Quantity:
    """Compute small-angle bright fringe position on a screen: ``y_m = m * lambda * L / d``.

    Args:
        order: Non-negative integer interference order ``m >= 0``.
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        screen_distance: Distance from slits to screen ``L`` (``[length]``, ``>= 0``).
        slit_separation: Center-to-center slit spacing ``d`` (``[length]``, ``> 0``).

    Returns:
        Bright fringe position as a ``pint.Quantity`` in ``meter``.
    """
    m = _validate_integer_order(order, min_value=0, name="order")
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    screen_l = validate_quantity(
        screen_distance,
        "[length]",
        name="screen_distance",
        allow_negative=False,
    )
    d = validate_quantity(
        slit_separation,
        "[length]",
        name="slit_separation",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit((m * lam * screen_l) / d, "m")


def double_slit_fringe_spacing(
    wavelength: QuantityLike,
    screen_distance: QuantityLike,
    slit_separation: QuantityLike,
) -> pint.Quantity:
    """Compute adjacent bright fringe spacing: ``delta_y = lambda * L / d``.

    Args:
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).
        screen_distance: Distance from slits to screen ``L`` (``[length]``, ``>= 0``).
        slit_separation: Center-to-center slit spacing ``d`` (``[length]``, ``> 0``).

    Returns:
        Fringe spacing as a ``pint.Quantity`` in ``meter``.
    """
    return double_slit_fringe_position(1, wavelength, screen_distance, slit_separation)


def thin_film_constructive_thickness(
    wavelength: QuantityLike,
    film_index: QuantityLike,
    *,
    order: QuantityLike = 0,
    phase_shifts: QuantityLike = 1,
) -> pint.Quantity:
    """Compute normal-incidence thin-film thickness for constructive reflection.

    Uses ``t = (m + 0.5) * lambda / (2 * n_film)`` when ``phase_shifts`` is odd
    (one relative ``pi`` phase inversion, e.g. soap film in air), and
    ``t = m * lambda / (2 * n_film)`` when ``phase_shifts`` is even (0 or 2).

    Args:
        wavelength: Vacuum light wavelength (dimension ``[length]``, ``>= 0``).
        film_index: Thin-film refractive index (dimensionless, ``> 0``).
        order: Non-negative integer interference order ``m >= 0`` (defaults to ``0``).
        phase_shifts: Number of ``pi`` reflection phase shifts across the two
            film boundaries (``0``, ``1``, or ``2``; defaults to ``1``).

    Returns:
        Film thickness as a ``pint.Quantity`` in ``meter``.

    Raises:
        PhysicalDomainError: If ``phase_shifts`` is not in ``{0, 1, 2}`` or
            inputs violate physical bounds.
    """
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    n_film = _validate_refractive_index(film_index, name="film_index")
    m = _validate_integer_order(order, min_value=0, name="order")
    shifts = _validate_integer_order(phase_shifts, min_value=0, name="phase_shifts")
    if shifts > 2:
        raise PhysicalDomainError("Parameter 'phase_shifts' must be 0, 1, or 2.")

    effective_order = m + 0.5 if shifts % 2 == 1 else float(m)
    return _to_clean_unit((effective_order * lam) / (2.0 * n_film), "m")


def michelson_displacement(
    fringe_count: QuantityLike,
    wavelength: QuantityLike,
) -> pint.Quantity:
    """Compute mirror displacement in a Michelson interferometer: ``delta_d = N * lambda / 2``.

    Args:
        fringe_count: Number of shifted fringes ``N`` (dimensionless, ``>= 0``).
        wavelength: Light wavelength (dimension ``[length]``, ``>= 0``).

    Returns:
        Mirror displacement as a ``pint.Quantity`` in ``meter``.
    """
    n_fringes = float(
        validate_quantity(
            fringe_count,
            "dimensionless",
            name="fringe_count",
            allow_negative=False,
        )
        .to("dimensionless")
        .magnitude
    )
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    return _to_clean_unit(0.5 * n_fringes * lam, "m")


__all__ = [
    "double_slit_angle",
    "double_slit_fringe_position",
    "double_slit_fringe_spacing",
    "michelson_displacement",
    "thin_film_constructive_thickness",
]
