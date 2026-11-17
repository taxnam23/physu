"""Temperature scale conversions and thermal expansion formulas."""

from __future__ import annotations

import math

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity

_ZERO_KELVIN_TOL = 1e-12


def _to_absolute_kelvin(
    temperature: QuantityLike,
    *,
    name: str = "temperature",
    allow_zero: bool = True,
) -> pint.Quantity:
    """Validate an absolute temperature (``>= 0 K`` or ``> 0 K``) and return it in ``kelvin``."""
    raw = validate_quantity(temperature, "[temperature]", name=name)
    if str(raw.units).startswith("delta_"):
        raise PhysicalDomainError(
            f"Parameter '{name}' requires an absolute temperature unit, got '{raw.units}'."
        )
    kelvin = to_unit(raw, "K")
    mag = float(kelvin.magnitude)
    if mag < 0.0:
        if math.isclose(mag, 0.0, abs_tol=_ZERO_KELVIN_TOL):
            kelvin = Quantity(0.0, "K")
            mag = 0.0
        else:
            raise PhysicalDomainError(
                f"Parameter '{name}' cannot be below absolute zero (0 K), got {kelvin}."
            )
    if not allow_zero and mag == 0.0:
        raise PhysicalDomainError(f"Parameter '{name}' must be strictly above 0 K.")
    return kelvin


def _to_delta_kelvin(
    delta_temperature: QuantityLike,
    *,
    name: str = "delta_temperature",
    allow_zero: bool = True,
) -> pint.Quantity:
    """Validate a temperature interval (``K``, ``delta_degC``, ``delta_degF``) in ``kelvin``."""
    raw = validate_quantity(
        delta_temperature,
        "[temperature]",
        name=name,
        allow_zero=allow_zero,
    )
    try:
        # Reject offset units (e.g. bare degC / degF) whose intervals are ambiguous.
        _ = raw * 1.0
    except pint.OffsetUnitCalculusError as exc:
        raise PhysicalDomainError(
            f"Parameter '{name}' cannot use ambiguous offset unit '{raw.units}'; "
            "use 'delta_degC', 'delta_degF', or 'K' (or temperature_difference)."
        ) from exc
    return to_unit(raw, "K")


def _thermal_expansion(
    initial_extent: QuantityLike,
    expansion_coefficient: QuantityLike,
    delta_temperature: QuantityLike,
    *,
    extent_dimension: str,
    extent_name: str,
) -> pint.Quantity:
    """Compute thermal expansion ``delta_X = coeff * X_0 * delta_T`` in SI base units."""
    x0 = validate_quantity(
        initial_extent,
        extent_dimension,
        name=extent_name,
        allow_negative=False,
    )
    coeff = validate_quantity(
        expansion_coefficient,
        "1 / K",
        name="expansion_coefficient",
    )
    dt = _to_delta_kelvin(delta_temperature, name="delta_temperature")
    return to_si(coeff * x0 * dt)


def to_kelvin(temperature: QuantityLike) -> pint.Quantity:
    """Convert an absolute temperature to ``kelvin``, enforcing ``T >= 0 K``.

    Args:
        temperature: Absolute temperature (e.g. ``"25 degC"``, ``"77 degF"``,
            ``"300 K"``).

    Returns:
        Temperature as a ``pint.Quantity`` in ``kelvin``.
    """
    return _to_absolute_kelvin(temperature, name="temperature", allow_zero=True)


def to_celsius(temperature: QuantityLike) -> pint.Quantity:
    """Convert an absolute temperature to ``degree_Celsius``, enforcing ``T >= 0 K``.

    Args:
        temperature: Absolute temperature (compatible with ``[temperature]``).

    Returns:
        Temperature as a ``pint.Quantity`` in ``degree_Celsius``.
    """
    return to_unit(to_kelvin(temperature), "degC")


def to_fahrenheit(temperature: QuantityLike) -> pint.Quantity:
    """Convert an absolute temperature to ``degree_Fahrenheit``, enforcing ``T >= 0 K``.

    Args:
        temperature: Absolute temperature (compatible with ``[temperature]``).

    Returns:
        Temperature as a ``pint.Quantity`` in ``degree_Fahrenheit``.
    """
    return to_unit(to_kelvin(temperature), "degF")


def temperature_difference(
    initial_temperature: QuantityLike,
    final_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute signed temperature change ``delta_T = T_f - T_i`` in ``kelvin``.

    Safely handles offset scales such as ``degC`` and ``degF`` by normalizing
    both endpoints to Kelvin before subtraction.

    Args:
        initial_temperature: Initial absolute temperature (``>= 0 K``).
        final_temperature: Final absolute temperature (``>= 0 K``).

    Returns:
        Temperature interval as a ``pint.Quantity`` in ``kelvin`` (convertible to
        ``delta_degC`` or ``delta_degF``).
    """
    t_i = _to_absolute_kelvin(initial_temperature, name="initial_temperature")
    t_f = _to_absolute_kelvin(final_temperature, name="final_temperature")
    return to_unit(t_f - t_i, "K")


def linear_thermal_expansion(
    initial_length: QuantityLike,
    expansion_coefficient: QuantityLike,
    delta_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute linear thermal expansion: ``delta_L = alpha * L_0 * delta_T``.

    Args:
        initial_length: Initial length (dimension ``[length]``, ``>= 0``).
        expansion_coefficient: Linear expansion coefficient (compatible with ``1 / K``).
        delta_temperature: Temperature interval (``K``, ``delta_degC``, or ``delta_degF``).

    Returns:
        Length change as an SI base ``pint.Quantity`` (``meter``).
    """
    return _thermal_expansion(
        initial_length,
        expansion_coefficient,
        delta_temperature,
        extent_dimension="[length]",
        extent_name="initial_length",
    )


def volumetric_thermal_expansion(
    initial_volume: QuantityLike,
    expansion_coefficient: QuantityLike,
    delta_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute volumetric thermal expansion: ``delta_V = beta * V_0 * delta_T``.

    Args:
        initial_volume: Initial volume (dimension ``[volume]``, ``>= 0``).
        expansion_coefficient: Volumetric expansion coefficient (compatible with ``1 / K``).
        delta_temperature: Temperature interval (``K``, ``delta_degC``, or ``delta_degF``).

    Returns:
        Volume change as an SI base ``pint.Quantity`` (``meter ** 3``).
    """
    return _thermal_expansion(
        initial_volume,
        expansion_coefficient,
        delta_temperature,
        extent_dimension="[volume]",
        extent_name="initial_volume",
    )


__all__ = [
    "linear_thermal_expansion",
    "temperature_difference",
    "to_celsius",
    "to_fahrenheit",
    "to_kelvin",
    "volumetric_thermal_expansion",
]
