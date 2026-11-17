"""Thermodynamic laws, process work, heat engine efficiency, and entropy formulas."""

from __future__ import annotations

import math

import pint

from ..constants import MOLAR_GAS_CONSTANT
from ..energy.conservation import mechanical_efficiency
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity
from .ideal_gas import _validate_positive_pressure, _validate_positive_volume
from .temperature import _to_absolute_kelvin


def internal_energy_change(
    heat: QuantityLike,
    work_done: QuantityLike,
) -> pint.Quantity:
    """Compute internal energy change via the first law: ``delta_U = Q - W``.

    Uses the standard physics convention where positive ``heat`` enters the
    system and positive ``work_done`` is work performed *by* the system.

    Args:
        heat: Net heat added to the system (dimension ``[energy]``).
        work_done: Work done by the system (dimension ``[energy]``).

    Returns:
        Internal energy change as a ``pint.Quantity`` in ``joule``.
    """
    q = validate_quantity(heat, "[energy]", name="heat")
    w = validate_quantity(work_done, "[energy]", name="work_done")
    return to_unit(q - w, "J")


def isobaric_work(
    pressure: QuantityLike,
    volume_change: QuantityLike,
) -> pint.Quantity:
    """Compute thermodynamic work at constant pressure: ``W = P * delta_V``.

    Args:
        pressure: Constant gas pressure (dimension ``[pressure]``, ``>= 0``).
        volume_change: Volume change ``V_f - V_i`` (dimension ``[volume]``).

    Returns:
        Work done by the gas as a ``pint.Quantity`` in ``joule``.
    """
    p = _validate_positive_pressure(pressure, allow_zero=True)
    dv = validate_quantity(volume_change, "[volume]", name="volume_change")
    result = to_unit(p * dv, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def isothermal_work(
    moles: QuantityLike,
    temperature: QuantityLike,
    initial_volume: QuantityLike,
    final_volume: QuantityLike,
) -> pint.Quantity:
    """Compute isothermal reversible ideal-gas work: ``W = n * R * T * ln(V_f / V_i)``.

    Args:
        moles: Amount of gas (dimension ``[substance]``, ``>= 0``).
        temperature: Constant absolute temperature (``>= 0 K``).
        initial_volume: Initial volume (dimension ``[volume]``, ``> 0``).
        final_volume: Final volume (dimension ``[volume]``, ``> 0``).

    Returns:
        Work done by the gas as a ``pint.Quantity`` in ``joule``.
    """
    n = validate_quantity(moles, "[substance]", name="moles", allow_negative=False)
    t = _to_absolute_kelvin(temperature, name="temperature")
    v_i = _validate_positive_volume(initial_volume, name="initial_volume")
    v_f = _validate_positive_volume(final_volume, name="final_volume")
    ratio = float(to_si(v_f / v_i).magnitude)
    result = to_unit(n * MOLAR_GAS_CONSTANT * t * math.log(ratio), "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def thermal_efficiency(
    work_output: QuantityLike,
    heat_input: QuantityLike,
) -> pint.Quantity:
    """Compute heat engine thermal efficiency: ``eta = W_out / Q_h``.

    Args:
        work_output: Net useful work output (dimension ``[energy]``, ``>= 0``).
        heat_input: Heat absorbed from the hot reservoir (dimension ``[energy]``,
            ``> 0``).

    Returns:
        Dimensionless thermal efficiency in ``[0, 1]`` (convertible to ``"%"``).
    """
    return mechanical_efficiency(work_output, heat_input)


def carnot_efficiency(
    cold_temperature: QuantityLike,
    hot_temperature: QuantityLike,
) -> pint.Quantity:
    """Compute maximum theoretical Carnot efficiency: ``eta_Carnot = 1 - T_c / T_h``.

    Args:
        cold_temperature: Cold reservoir absolute temperature (``>= 0 K``).
        hot_temperature: Hot reservoir absolute temperature (``> 0 K``).

    Returns:
        Dimensionless Carnot efficiency in ``[0, 1]`` (convertible to ``"%"``).

    Raises:
        PhysicalDomainError: If ``cold_temperature > hot_temperature`` or inputs
            violate absolute-temperature bounds.
    """
    t_c = _to_absolute_kelvin(cold_temperature, name="cold_temperature")
    t_h = _to_absolute_kelvin(hot_temperature, name="hot_temperature", allow_zero=False)
    if float(t_c.magnitude) > float(t_h.magnitude):
        raise PhysicalDomainError(
            f"Cold reservoir temperature ({t_c}) cannot exceed hot reservoir "
            f"temperature ({t_h})."
        )
    return to_si(1.0 - (t_c / t_h))


def isothermal_entropy_change(
    reversible_heat: QuantityLike,
    temperature: QuantityLike,
) -> pint.Quantity:
    """Compute entropy change for an isothermal reversible process: ``delta_S = Q_rev / T``.

    Args:
        reversible_heat: Heat transferred reversibly (dimension ``[energy]``).
        temperature: Constant absolute temperature (``> 0 K``).

    Returns:
        Entropy change as a ``pint.Quantity`` in ``joule / kelvin``.
    """
    q = validate_quantity(reversible_heat, "[energy]", name="reversible_heat")
    t = _to_absolute_kelvin(temperature, name="temperature", allow_zero=False)
    result = to_unit(q / t, "J / K")
    return Quantity(0.0, "J / K") if result.magnitude == 0 else result


__all__ = [
    "carnot_efficiency",
    "internal_energy_change",
    "isobaric_work",
    "isothermal_entropy_change",
    "isothermal_work",
    "thermal_efficiency",
]
