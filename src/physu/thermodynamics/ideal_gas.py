"""Ideal gas law and kinetic theory of gases formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..constants import BOLTZMANN_CONSTANT, MOLAR_GAS_CONSTANT
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity
from .temperature import _to_absolute_kelvin


def _validate_positive_pressure(
    pressure: QuantityLike,
    *,
    name: str = "pressure",
    allow_zero: bool = False,
) -> pint.Quantity:
    """Validate non-negative or strictly positive pressure."""
    return validate_quantity(
        pressure,
        "[pressure]",
        name=name,
        allow_negative=False,
        allow_zero=allow_zero,
    )


def _validate_positive_volume(
    volume: QuantityLike,
    *,
    name: str = "volume",
    allow_zero: bool = False,
) -> pint.Quantity:
    """Validate non-negative or strictly positive volume."""
    return validate_quantity(
        volume,
        "[volume]",
        name=name,
        allow_negative=False,
        allow_zero=allow_zero,
    )


def ideal_gas_pressure(
    moles: QuantityLike,
    temperature: QuantityLike,
    volume: QuantityLike,
) -> pint.Quantity:
    """Compute pressure from the ideal gas law: ``P = n * R * T / V``.

    Args:
        moles: Amount of substance (dimension ``[substance]``, ``>= 0``).
        temperature: Absolute temperature (``>= 0 K``).
        volume: Container volume (dimension ``[volume]``, ``> 0``).

    Returns:
        Pressure as a ``pint.Quantity`` in ``pascal``.
    """
    n = validate_quantity(moles, "[substance]", name="moles", allow_negative=False)
    t = _to_absolute_kelvin(temperature, name="temperature")
    v = _validate_positive_volume(volume)
    return to_unit(n * MOLAR_GAS_CONSTANT * t / v, "Pa")


def ideal_gas_volume(
    moles: QuantityLike,
    temperature: QuantityLike,
    pressure: QuantityLike,
) -> pint.Quantity:
    """Compute volume from the ideal gas law: ``V = n * R * T / P``.

    Args:
        moles: Amount of substance (dimension ``[substance]``, ``>= 0``).
        temperature: Absolute temperature (``>= 0 K``).
        pressure: Gas pressure (dimension ``[pressure]``, ``> 0``).

    Returns:
        Volume as an SI base ``pint.Quantity`` (``meter ** 3``).
    """
    n = validate_quantity(moles, "[substance]", name="moles", allow_negative=False)
    t = _to_absolute_kelvin(temperature, name="temperature")
    p = _validate_positive_pressure(pressure)
    return to_si(n * MOLAR_GAS_CONSTANT * t / p)


def ideal_gas_moles(
    pressure: QuantityLike,
    volume: QuantityLike,
    temperature: QuantityLike,
) -> pint.Quantity:
    """Compute amount of substance from the ideal gas law: ``n = P * V / (R * T)``.

    Args:
        pressure: Gas pressure (dimension ``[pressure]``, ``>= 0``).
        volume: Container volume (dimension ``[volume]``, ``>= 0``).
        temperature: Absolute temperature (``> 0 K``).

    Returns:
        Amount of substance as an SI base ``pint.Quantity`` (``mole``).
    """
    p = _validate_positive_pressure(pressure, allow_zero=True)
    v = _validate_positive_volume(volume, allow_zero=True)
    t = _to_absolute_kelvin(temperature, name="temperature", allow_zero=False)
    return to_si((p * v) / (MOLAR_GAS_CONSTANT * t))


def ideal_gas_temperature(
    pressure: QuantityLike,
    volume: QuantityLike,
    moles: QuantityLike,
) -> pint.Quantity:
    """Compute absolute temperature from the ideal gas law: ``T = P * V / (n * R)``.

    Args:
        pressure: Gas pressure (dimension ``[pressure]``, ``>= 0``).
        volume: Container volume (dimension ``[volume]``, ``>= 0``).
        moles: Amount of substance (dimension ``[substance]``, ``> 0``).

    Returns:
        Absolute temperature as a ``pint.Quantity`` in ``kelvin``.
    """
    p = _validate_positive_pressure(pressure, allow_zero=True)
    v = _validate_positive_volume(volume, allow_zero=True)
    n = validate_quantity(
        moles,
        "[substance]",
        name="moles",
        allow_negative=False,
        allow_zero=False,
    )
    return to_unit((p * v) / (n * MOLAR_GAS_CONSTANT), "K")


def combined_gas_law(
    initial_pressure: QuantityLike,
    initial_volume: QuantityLike,
    initial_temperature: QuantityLike,
    *,
    final_pressure: QuantityLike | None = None,
    final_volume: QuantityLike | None = None,
    final_temperature: QuantityLike | None = None,
) -> pint.Quantity:
    """Solve for one missing final state variable using ``P1 * V1 / T1 = P2 * V2 / T2``.

    Provide exactly two of ``final_pressure``, ``final_volume``, and ``final_temperature``.

    Args:
        initial_pressure: Initial pressure (``[pressure]``, ``> 0``).
        initial_volume: Initial volume (``[volume]``, ``> 0``).
        initial_temperature: Initial absolute temperature (``> 0 K``).
        final_pressure: Final pressure (``[pressure]``).
        final_volume: Final volume (``[volume]``).
        final_temperature: Final absolute temperature (``>= 0 K``).

    Returns:
        The missing final state quantity in SI units (``pascal``, ``meter ** 3``,
        or ``kelvin``).

    Raises:
        ValueError: If the number of provided final state arguments is not two.
    """
    missing_count = sum(
        arg is None for arg in (final_pressure, final_volume, final_temperature)
    )
    if missing_count != 1:
        raise ValueError(
            "Provide exactly two of 'final_pressure', 'final_volume', and "
            "'final_temperature'."
        )

    p1 = _validate_positive_pressure(initial_pressure, name="initial_pressure")
    v1 = _validate_positive_volume(initial_volume, name="initial_volume")
    t1 = _to_absolute_kelvin(
        initial_temperature,
        name="initial_temperature",
        allow_zero=False,
    )

    if final_pressure is None:
        assert final_volume is not None and final_temperature is not None
        v2 = _validate_positive_volume(final_volume, name="final_volume")
        t2 = _to_absolute_kelvin(final_temperature, name="final_temperature")
        return to_unit(p1 * (v1 / v2) * (t2 / t1), "Pa")

    if final_volume is None:
        assert final_pressure is not None and final_temperature is not None
        p2 = _validate_positive_pressure(final_pressure, name="final_pressure")
        t2 = _to_absolute_kelvin(final_temperature, name="final_temperature")
        return to_si(v1 * (p1 / p2) * (t2 / t1))

    assert final_pressure is not None and final_volume is not None
    p2 = _validate_positive_pressure(
        final_pressure,
        name="final_pressure",
        allow_zero=True,
    )
    v2 = _validate_positive_volume(final_volume, name="final_volume", allow_zero=True)
    return to_unit(t1 * (p2 * v2) / (p1 * v1), "K")


def rms_molecular_speed(
    temperature: QuantityLike,
    molar_mass: QuantityLike,
) -> pint.Quantity:
    """Compute root-mean-square speed of ideal gas molecules: ``v_rms = sqrt(3 * R * T / M)``.

    Args:
        temperature: Absolute gas temperature (``>= 0 K``).
        molar_mass: Molar mass (compatible with ``kg / mol``, e.g. ``"28 g / mol"``,
            ``> 0``).

    Returns:
        RMS molecular speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    t = _to_absolute_kelvin(temperature, name="temperature")
    m_mol = validate_quantity(
        molar_mass,
        "kg / mol",
        name="molar_mass",
        allow_negative=False,
        allow_zero=False,
    )
    speed_sq = float(to_si(3.0 * MOLAR_GAS_CONSTANT * t / m_mol).magnitude)
    return Quantity(math.sqrt(speed_sq), "m / s")


def average_molecular_kinetic_energy(temperature: QuantityLike) -> pint.Quantity:
    """Compute average translational kinetic energy per gas molecule: ``K_avg = 1.5 * k_B * T``.

    Args:
        temperature: Absolute gas temperature (``>= 0 K``).

    Returns:
        Average molecular translational kinetic energy as a ``pint.Quantity`` in
        ``joule``.
    """
    t = _to_absolute_kelvin(temperature, name="temperature")
    return to_unit(1.5 * BOLTZMANN_CONSTANT * t, "J")


__all__ = [
    "average_molecular_kinetic_energy",
    "combined_gas_law",
    "ideal_gas_moles",
    "ideal_gas_pressure",
    "ideal_gas_temperature",
    "ideal_gas_volume",
    "rms_molecular_speed",
]
