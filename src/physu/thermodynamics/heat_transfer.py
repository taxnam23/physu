"""Conduction and thermal radiation heat-transfer formulas."""

from __future__ import annotations

import pint

from ..constants import STEFAN_BOLTZMANN_CONSTANT
from ..units import Quantity, QuantityLike, to_unit
from ..validation import require_range, validate_quantity
from .temperature import _to_absolute_kelvin, _to_delta_kelvin


def conduction_heat_rate(
    thermal_conductivity: QuantityLike,
    area: QuantityLike,
    delta_temperature: QuantityLike,
    thickness: QuantityLike,
) -> pint.Quantity:
    """Compute steady-state conductive heat transfer rate: ``Q_dot = k * A * delta_T / L``.

    Args:
        thermal_conductivity: Material thermal conductivity (compatible with
            ``W / (m * K)``, ``>= 0``).
        area: Cross-sectional area perpendicular to heat flow (``[area]``, ``>= 0``).
        delta_temperature: Temperature difference across material (``K``,
            ``delta_degC``, or ``delta_degF``).
        thickness: Conduction path thickness (dimension ``[length]``, ``> 0``).

    Returns:
        Heat transfer rate as a ``pint.Quantity`` in ``watt``.
    """
    k = validate_quantity(
        thermal_conductivity,
        "W / (m * K)",
        name="thermal_conductivity",
        allow_negative=False,
    )
    a = validate_quantity(area, "[area]", name="area", allow_negative=False)
    dt = _to_delta_kelvin(delta_temperature, name="delta_temperature")
    length = validate_quantity(
        thickness,
        "[length]",
        name="thickness",
        allow_negative=False,
        allow_zero=False,
    )
    result = to_unit(k * a * dt / length, "W")
    return Quantity(0.0, "W") if result.magnitude == 0 else result


def radiation_power(
    emissivity: QuantityLike,
    area: QuantityLike,
    temperature: QuantityLike,
    *,
    ambient_temperature: QuantityLike | None = None,
    stefan_boltzmann_constant: QuantityLike = STEFAN_BOLTZMANN_CONSTANT,
) -> pint.Quantity:
    """Compute radiated thermal power via Stefan-Boltzmann law.

    Computes ``P = emissivity * sigma * A * T^4`` when ``ambient_temperature`` is
    omitted, or net radiative heat transfer ``P_net = emissivity * sigma * A * (T^4 - T_amb^4)``
    when ``ambient_temperature`` is provided.

    Args:
        emissivity: Surface emissivity in ``[0, 1]`` (dimensionless).
        area: Radiating surface area (dimension ``[area]``, ``>= 0``).
        temperature: Radiating surface absolute temperature (``>= 0 K``).
        ambient_temperature: Optional surrounding absolute temperature (``>= 0 K``).
        stefan_boltzmann_constant: Stefan-Boltzmann constant (compatible with
            ``W / (m^2 * K^4)``, ``> 0``, defaults to ``STEFAN_BOLTZMANN_CONSTANT``).

    Returns:
        Radiated power as a ``pint.Quantity`` in ``watt``.
    """
    eps = validate_quantity(emissivity, "dimensionless", name="emissivity")
    require_range(
        eps,
        min_value=Quantity(0.0, "dimensionless"),
        max_value=Quantity(1.0, "dimensionless"),
        name="emissivity",
    )
    a = validate_quantity(area, "[area]", name="area", allow_negative=False)
    t = _to_absolute_kelvin(temperature, name="temperature")
    sigma = validate_quantity(
        stefan_boltzmann_constant,
        "W / (m^2 * K^4)",
        name="stefan_boltzmann_constant",
        allow_negative=False,
        allow_zero=False,
    )
    if ambient_temperature is None:
        temp_factor = to_unit(t**4, "K^4")
    else:
        t_amb = _to_absolute_kelvin(ambient_temperature, name="ambient_temperature")
        temp_factor = to_unit((t**4) - (t_amb**4), "K^4")

    result = to_unit(eps * sigma * a * temp_factor, "W")
    return Quantity(0.0, "W") if result.magnitude == 0 else result


__all__ = [
    "conduction_heat_rate",
    "radiation_power",
]
