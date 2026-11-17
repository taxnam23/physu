"""Calorimetry formulas for sensible heat, latent heat, and thermal equilibrium."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity
from .temperature import _to_absolute_kelvin, _to_delta_kelvin


def sensible_heat(
    mass: QuantityLike,
    specific_heat: QuantityLike,
    delta_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute heat transferred during a temperature change: ``Q = m * c * delta_T``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        specific_heat: Specific heat capacity (compatible with ``J / (kg * K)``, ``>= 0``).
        delta_temperature: Temperature interval (``K``, ``delta_degC``, or ``delta_degF``).

    Returns:
        Heat transferred as a ``pint.Quantity`` in ``joule``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    c = validate_quantity(
        specific_heat,
        "J / (kg * K)",
        name="specific_heat",
        allow_negative=False,
    )
    dt = _to_delta_kelvin(delta_temperature, name="delta_temperature")
    result = to_unit(m * c * dt, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def latent_heat(
    mass: QuantityLike,
    specific_latent_heat: QuantityLike,
) -> pint.Quantity:
    """Compute thermal energy for a phase transition: ``Q = m * L``.

    Args:
        mass: Substance mass undergoing phase change (dimension ``[mass]``, ``>= 0``).
        specific_latent_heat: Specific latent heat (compatible with ``J / kg``, ``>= 0``).

    Returns:
        Latent heat magnitude as a ``pint.Quantity`` in ``joule``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    l_val = validate_quantity(
        specific_latent_heat,
        "J / kg",
        name="specific_latent_heat",
        allow_negative=False,
    )
    return to_unit(m * l_val, "J")


def specific_heat_capacity(
    heat: QuantityLike,
    mass: QuantityLike,
    delta_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute specific heat capacity: ``c = Q / (m * delta_T)``.

    Args:
        heat: Heat energy transferred (dimension ``[energy]``).
        mass: Sample mass (dimension ``[mass]``, ``> 0``).
        delta_temperature: Non-zero temperature interval (``K``, ``delta_degC``, or
            ``delta_degF``).

    Returns:
        Specific heat capacity as a ``pint.Quantity`` in ``joule / (kilogram * kelvin)``.

    Raises:
        PhysicalDomainError: If ``heat`` and ``delta_temperature`` have opposite signs.
    """
    q = validate_quantity(heat, "[energy]", name="heat")
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False, allow_zero=False)
    dt = _to_delta_kelvin(delta_temperature, name="delta_temperature", allow_zero=False)
    computed = to_unit(q / (m * dt), "J / (kg * K)")
    if float(computed.magnitude) < 0.0:
        raise PhysicalDomainError(
            "Heat and temperature change have opposite signs, yielding a non-physical "
            f"negative specific heat capacity ({computed})."
        )
    return computed


def equilibrium_temperature(
    bodies: Sequence[tuple[QuantityLike, QuantityLike, QuantityLike]],
) -> pint.Quantity:
    """Compute final thermal equilibrium temperature of mixed bodies without phase change.

    Solves ``T_eq = sum(m_i * c_i * T_i) / sum(m_i * c_i)``.

    Args:
        bodies: Non-empty sequence of ``(mass, specific_heat, initial_temperature)``
            tuples, where each ``mass >= 0``, ``specific_heat >= 0``, and
            ``initial_temperature >= 0 K``.

    Returns:
        Equilibrium temperature as a ``pint.Quantity`` in ``kelvin`` (convertible
        to ``degC`` or ``degF``).
    """
    if len(bodies) == 0:
        raise PhysicalDomainError("At least one body must be provided.")

    weighted_temps: list[float] = []
    heat_capacities: list[float] = []
    for idx, entry in enumerate(bodies):
        if len(entry) != 3:
            raise ValueError(
                f"Expected a 3-tuple (mass, specific_heat, temperature) at bodies[{idx}]."
            )
        mass_val, c_val, temp_val = entry
        m = float(
            validate_quantity(
                mass_val,
                "[mass]",
                name=f"bodies[{idx}].mass",
                allow_negative=False,
            )
            .to("kg")
            .magnitude
        )
        c = float(
            validate_quantity(
                c_val,
                "J / (kg * K)",
                name=f"bodies[{idx}].specific_heat",
                allow_negative=False,
            )
            .to("J / (kg * K)")
            .magnitude
        )
        t = float(
            _to_absolute_kelvin(
                temp_val,
                name=f"bodies[{idx}].temperature",
            ).magnitude
        )
        cap = m * c
        heat_capacities.append(cap)
        weighted_temps.append(cap * t)

    total_capacity = math.fsum(heat_capacities)
    if total_capacity <= 0.0:
        raise PhysicalDomainError(
            "Total heat capacity across bodies must be strictly positive."
        )
    return Quantity(math.fsum(weighted_temps) / total_capacity, "K")


__all__ = [
    "equilibrium_temperature",
    "latent_heat",
    "sensible_heat",
    "specific_heat_capacity",
]
