"""Capacitance formulas: capacitors, dielectrics, stored energy, and RC transients."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pint

from ..constants import VACUUM_PERMITTIVITY
from ..exceptions import PhysicalDomainError
from ..units import QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import (
    _combine_additive,
    _combine_reciprocal,
    _to_clean_unit,
    _validate_positive_factor,
)


def _rc_decay_factor(
    time: QuantityLike,
    resistance: QuantityLike,
    capacitance_value: QuantityLike,
) -> float:
    """Validate positive R, C and non-negative time, returning ``exp(-t / (R * C))``."""
    t = validate_quantity(time, "[time]", name="time", allow_negative=False)
    r = validate_quantity(
        resistance,
        "[resistance]",
        name="resistance",
        allow_negative=False,
        allow_zero=False,
    )
    c = validate_quantity(
        capacitance_value,
        "[capacitance]",
        name="capacitance",
        allow_negative=False,
        allow_zero=False,
    )
    tau = rc_time_constant(r, c)
    ratio = float(to_si(t / tau).magnitude)
    return math.exp(-ratio)


def capacitance(
    charge: QuantityLike,
    voltage: QuantityLike,
) -> pint.Quantity:
    """Compute capacitance from stored charge and voltage: ``C = Q / V``.

    Args:
        charge: Stored electric charge (dimension ``[charge]``).
        voltage: Non-zero potential difference across plates (dimension
            ``[electric_potential]``, ``!= 0``).

    Returns:
        Capacitance as a ``pint.Quantity`` in ``farad``.

    Raises:
        PhysicalDomainError: If ``charge`` and ``voltage`` have opposite signs
            (implying negative capacitance) or ``voltage == 0``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    v = validate_quantity(
        voltage,
        "[electric_potential]",
        name="voltage",
        allow_zero=False,
    )
    result = _to_clean_unit(q / v, "F")
    if float(result.magnitude) < 0:
        raise PhysicalDomainError(
            "Capacitance cannot be negative; charge and voltage must have matching signs."
        )
    return result


def capacitor_charge(
    capacitance: QuantityLike,
    voltage: QuantityLike,
) -> pint.Quantity:
    """Compute electric charge stored on a capacitor: ``Q = C * V``.

    Args:
        capacitance: Capacitance (dimension ``[capacitance]``, ``>= 0``).
        voltage: Potential difference across the capacitor (dimension
            ``[electric_potential]``).

    Returns:
        Electric charge as a ``pint.Quantity`` in ``coulomb``.
    """
    c = validate_quantity(
        capacitance,
        "[capacitance]",
        name="capacitance",
        allow_negative=False,
    )
    v = validate_quantity(voltage, "[electric_potential]", name="voltage")
    return _to_clean_unit(c * v, "C")


def capacitor_voltage(
    charge: QuantityLike,
    capacitance: QuantityLike,
) -> pint.Quantity:
    """Compute voltage across a capacitor: ``V = Q / C``.

    Args:
        charge: Stored electric charge (dimension ``[charge]``).
        capacitance: Positive capacitance (dimension ``[capacitance]``, ``> 0``).

    Returns:
        Potential difference as a ``pint.Quantity`` in ``volt``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    c = validate_quantity(
        capacitance,
        "[capacitance]",
        name="capacitance",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(q / c, "V")


def parallel_plate_capacitance(
    area: QuantityLike,
    separation: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute parallel-plate capacitance: ``C = eps_0 * eps_r * A / d``.

    Args:
        area: Overlapping plate area (dimension ``[area]``, ``>= 0``).
        separation: Plate separation distance (dimension ``[length]``, ``> 0``).
        relative_permittivity: Dielectric relative permittivity (dimensionless,
            ``> 0``, defaults to ``1.0``).

    Returns:
        Capacitance as a ``pint.Quantity`` in ``farad``.
    """
    area_q = validate_quantity(area, "[area]", name="area", allow_negative=False)
    d = validate_quantity(
        separation,
        "[length]",
        name="separation",
        allow_negative=False,
        allow_zero=False,
    )
    eps_r = _validate_positive_factor(
        relative_permittivity,
        name="relative_permittivity",
    )
    return _to_clean_unit(VACUUM_PERMITTIVITY * eps_r * area_q / d, "F")


def capacitor_energy(
    *,
    capacitance: QuantityLike | None = None,
    voltage: QuantityLike | None = None,
    charge: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute electrostatic energy stored in a capacitor.

    Uses ``U = 0.5 * C * V^2``, ``U = Q^2 / (2 * C)``, or ``U = 0.5 * Q * V``.
    Provide exactly two of ``capacitance``, ``voltage``, or ``charge``.

    Args:
        capacitance: Capacitance (dimension ``[capacitance]``, ``>= 0`` when paired
            with ``voltage``, ``> 0`` when paired with ``charge``).
        voltage: Potential difference (dimension ``[electric_potential]``).
        charge: Stored charge (dimension ``[charge]``).

    Returns:
        Stored electrostatic energy as a ``pint.Quantity`` in ``joule``.

    Raises:
        ValueError: If fewer or more than two parameters are provided.
        PhysicalDomainError: If ``charge`` and ``voltage`` have opposite signs.
    """
    provided = (capacitance is not None) + (voltage is not None) + (charge is not None)
    if provided != 2:
        raise ValueError(
            "Provide exactly two of 'capacitance', 'voltage', or 'charge'."
        )

    c = (
        validate_quantity(
            capacitance,
            "[capacitance]",
            name="capacitance",
            allow_negative=False,
            allow_zero=(charge is None),
        )
        if capacitance is not None
        else None
    )
    v = (
        validate_quantity(voltage, "[electric_potential]", name="voltage")
        if voltage is not None
        else None
    )
    q = (
        validate_quantity(charge, "[charge]", name="charge")
        if charge is not None
        else None
    )

    if c is not None and v is not None:
        return _to_clean_unit(0.5 * c * (v**2), "J")
    if c is not None and q is not None:
        return _to_clean_unit((q**2) / (2.0 * c), "J")
    assert q is not None and v is not None
    energy = _to_clean_unit(0.5 * q * v, "J")
    if float(energy.magnitude) < 0:
        raise PhysicalDomainError(
            "Stored capacitor energy cannot be negative; "
            "charge and voltage must have matching signs."
        )
    return energy


def series_capacitance(capacitances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent capacitance in series: ``1 / C_eq = sum(1 / C_i)``.

    Args:
        capacitances: Non-empty sequence of positive capacitances
            (dimension ``[capacitance]``, each ``> 0``).

    Returns:
        Equivalent capacitance as a ``pint.Quantity`` in ``farad``.
    """
    return _combine_reciprocal(
        capacitances,
        "[capacitance]",
        "F",
        item_label="capacitances",
    )


def parallel_capacitance(capacitances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent capacitance in parallel: ``C_eq = sum(C_i)``.

    Args:
        capacitances: Non-empty sequence of non-negative capacitances
            (dimension ``[capacitance]``, each ``>= 0``).

    Returns:
        Equivalent capacitance as a ``pint.Quantity`` in ``farad``.
    """
    return _combine_additive(
        capacitances,
        "[capacitance]",
        "F",
        item_label="capacitances",
        allow_zero=True,
    )


def rc_time_constant(
    resistance: QuantityLike,
    capacitance: QuantityLike,
) -> pint.Quantity:
    """Compute RC circuit time constant: ``tau = R * C``.

    Args:
        resistance: Circuit resistance (dimension ``[resistance]``, ``>= 0``).
        capacitance: Circuit capacitance (dimension ``[capacitance]``, ``>= 0``).

    Returns:
        Time constant as a ``pint.Quantity`` in ``second``.
    """
    r = validate_quantity(
        resistance,
        "[resistance]",
        name="resistance",
        allow_negative=False,
    )
    c = validate_quantity(
        capacitance,
        "[capacitance]",
        name="capacitance",
        allow_negative=False,
    )
    return _to_clean_unit(r * c, "s")


def rc_charging_voltage(
    source_voltage: QuantityLike,
    time: QuantityLike,
    resistance: QuantityLike,
    capacitance: QuantityLike,
) -> pint.Quantity:
    """Compute capacitor voltage during RC charging: ``V(t) = V_0 * (1 - exp(-t / (R * C)))``.

    Args:
        source_voltage: Supply voltage (dimension ``[electric_potential]``).
        time: Elapsed charging time from zero initial voltage (``[time]``, ``>= 0``).
        resistance: Series resistance (dimension ``[resistance]``, ``> 0``).
        capacitance: Capacitance (dimension ``[capacitance]``, ``> 0``).

    Returns:
        Capacitor voltage at ``time`` as a ``pint.Quantity`` in ``volt``.
    """
    v0 = validate_quantity(source_voltage, "[electric_potential]", name="source_voltage")
    decay = _rc_decay_factor(time, resistance, capacitance)
    return _to_clean_unit(v0 * (1.0 - decay), "V")


def rc_discharging_voltage(
    initial_voltage: QuantityLike,
    time: QuantityLike,
    resistance: QuantityLike,
    capacitance: QuantityLike,
) -> pint.Quantity:
    """Compute capacitor voltage during RC discharging: ``V(t) = V_0 * exp(-t / (R * C))``.

    Args:
        initial_voltage: Initial capacitor voltage at ``t = 0``
            (dimension ``[electric_potential]``).
        time: Elapsed discharging time (``[time]``, ``>= 0``).
        resistance: Discharge resistance (dimension ``[resistance]``, ``> 0``).
        capacitance: Capacitance (dimension ``[capacitance]``, ``> 0``).

    Returns:
        Capacitor voltage at ``time`` as a ``pint.Quantity`` in ``volt``.
    """
    v0 = validate_quantity(initial_voltage, "[electric_potential]", name="initial_voltage")
    decay = _rc_decay_factor(time, resistance, capacitance)
    return _to_clean_unit(v0 * decay, "V")


__all__ = [
    "capacitance",
    "capacitor_charge",
    "capacitor_energy",
    "capacitor_voltage",
    "parallel_capacitance",
    "parallel_plate_capacitance",
    "rc_charging_voltage",
    "rc_discharging_voltage",
    "rc_time_constant",
    "series_capacitance",
]
