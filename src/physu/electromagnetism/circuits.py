"""DC circuit formulas: Ohm's law, wire resistivity, electric power, and resistor networks."""

from __future__ import annotations

from collections.abc import Sequence

import pint

from ..energy.power import work_from_power
from ..exceptions import PhysicalDomainError
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _combine_additive, _combine_reciprocal, _to_clean_unit


def ohms_law_voltage(
    current: QuantityLike,
    resistance: QuantityLike,
) -> pint.Quantity:
    """Compute voltage across a resistor via Ohm's law: ``V = I * R``.

    Args:
        current: Electric current (dimension ``[current]``).
        resistance: Electrical resistance (dimension ``[resistance]``, ``>= 0``).

    Returns:
        Voltage as a ``pint.Quantity`` in ``volt``.
    """
    i = validate_quantity(current, "[current]", name="current")
    r = validate_quantity(
        resistance,
        "[resistance]",
        name="resistance",
        allow_negative=False,
    )
    return _to_clean_unit(i * r, "V")


def ohms_law_current(
    voltage: QuantityLike,
    resistance: QuantityLike,
) -> pint.Quantity:
    """Compute electric current through a resistor via Ohm's law: ``I = V / R``.

    Args:
        voltage: Applied voltage (dimension ``[electric_potential]``).
        resistance: Electrical resistance (dimension ``[resistance]``, ``> 0``).

    Returns:
        Electric current as a ``pint.Quantity`` in ``ampere``.
    """
    v = validate_quantity(voltage, "[electric_potential]", name="voltage")
    r = validate_quantity(
        resistance,
        "[resistance]",
        name="resistance",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(v / r, "A")


def ohms_law_resistance(
    voltage: QuantityLike,
    current: QuantityLike,
) -> pint.Quantity:
    """Compute ohmic resistance via Ohm's law: ``R = V / I``.

    Args:
        voltage: Applied voltage (dimension ``[electric_potential]``).
        current: Non-zero electric current (dimension ``[current]``, ``!= 0``).

    Returns:
        Electrical resistance as a ``pint.Quantity`` in ``ohm``.

    Raises:
        PhysicalDomainError: If ``voltage`` and ``current`` have opposite signs
            (which would imply negative ohmic resistance) or ``current == 0``.
    """
    v = validate_quantity(voltage, "[electric_potential]", name="voltage")
    i = validate_quantity(current, "[current]", name="current", allow_zero=False)
    resistance = _to_clean_unit(v / i, "ohm")
    if float(resistance.magnitude) < 0:
        raise PhysicalDomainError(
            "Ohmic resistance cannot be negative; voltage and current must have matching signs."
        )
    return resistance


def wire_resistance(
    resistivity: QuantityLike,
    length: QuantityLike,
    cross_sectional_area: QuantityLike,
) -> pint.Quantity:
    """Compute electrical resistance of a uniform conductor: ``R = rho * L / A``.

    Args:
        resistivity: Material electrical resistivity (compatible with
            ``ohm * m``, ``>= 0``).
        length: Conductor length (dimension ``[length]``, ``>= 0``).
        cross_sectional_area: Cross-sectional area (dimension ``[area]``, ``> 0``).

    Returns:
        Conductor resistance as a ``pint.Quantity`` in ``ohm``.
    """
    rho = validate_quantity(
        resistivity,
        "ohm * m",
        name="resistivity",
        allow_negative=False,
    )
    length_q = validate_quantity(length, "[length]", name="length", allow_negative=False)
    area_q = validate_quantity(
        cross_sectional_area,
        "[area]",
        name="cross_sectional_area",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(rho * length_q / area_q, "ohm")


def electric_power(
    *,
    voltage: QuantityLike | None = None,
    current: QuantityLike | None = None,
    resistance: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute DC electric power: ``P = V * I``, ``P = I^2 * R``, or ``P = V^2 / R``.

    Provide exactly two of ``voltage``, ``current``, or ``resistance``.

    Args:
        voltage: Electric potential difference (dimension ``[electric_potential]``).
        current: Electric current (dimension ``[current]``).
        resistance: Electrical resistance (dimension ``[resistance]``, ``>= 0`` when
            paired with ``current``, ``> 0`` when paired with ``voltage``).

    Returns:
        Electric power as a ``pint.Quantity`` in ``watt``.

    Raises:
        ValueError: If fewer or more than two parameters are provided.
    """
    provided = (voltage is not None) + (current is not None) + (resistance is not None)
    if provided != 2:
        raise ValueError(
            "Provide exactly two of 'voltage', 'current', or 'resistance'."
        )

    v = (
        validate_quantity(voltage, "[electric_potential]", name="voltage")
        if voltage is not None
        else None
    )
    i = (
        validate_quantity(current, "[current]", name="current")
        if current is not None
        else None
    )
    r = (
        validate_quantity(
            resistance,
            "[resistance]",
            name="resistance",
            allow_negative=False,
            allow_zero=(v is None),
        )
        if resistance is not None
        else None
    )

    if v is not None and i is not None:
        return _to_clean_unit(v * i, "W")
    if i is not None and r is not None:
        return _to_clean_unit((i**2) * r, "W")
    assert v is not None and r is not None
    return _to_clean_unit((v**2) / r, "W")


def joule_heating(
    current: QuantityLike,
    resistance: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute thermal energy dissipated in a resistor: ``Q = I^2 * R * t``.

    Args:
        current: Electric current (dimension ``[current]``).
        resistance: Electrical resistance (dimension ``[resistance]``, ``>= 0``).
        time: Elapsed time interval (dimension ``[time]``, ``>= 0``).

    Returns:
        Dissipated energy as a ``pint.Quantity`` in ``joule``.
    """
    power = electric_power(current=current, resistance=resistance)
    return work_from_power(power, time)


def series_resistance(resistances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent resistance for resistors in series: ``R_eq = sum(R_i)``.

    Args:
        resistances: Non-empty sequence of non-negative resistances
            (dimension ``[resistance]``, each ``>= 0``).

    Returns:
        Equivalent resistance as a ``pint.Quantity`` in ``ohm``.
    """
    return _combine_additive(
        resistances,
        "[resistance]",
        "ohm",
        item_label="resistances",
        allow_zero=True,
    )


def parallel_resistance(resistances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent resistance for resistors in parallel: ``1 / R_eq = sum(1 / R_i)``.

    Args:
        resistances: Non-empty sequence of positive resistances
            (dimension ``[resistance]``, each ``> 0``).

    Returns:
        Equivalent resistance as a ``pint.Quantity`` in ``ohm``.
    """
    return _combine_reciprocal(
        resistances,
        "[resistance]",
        "ohm",
        item_label="resistances",
    )


def voltage_divider(
    input_voltage: QuantityLike,
    resistance_1: QuantityLike,
    resistance_2: QuantityLike,
) -> pint.Quantity:
    """Compute unloaded voltage divider output: ``V_out = V_in * R2 / (R1 + R2)``.

    Args:
        input_voltage: Source voltage (dimension ``[electric_potential]``).
        resistance_1: Upper series resistor (dimension ``[resistance]``, ``>= 0``).
        resistance_2: Lower resistor across which output is measured
            (dimension ``[resistance]``, ``>= 0``).

    Returns:
        Output voltage as a ``pint.Quantity`` in ``volt``.

    Raises:
        PhysicalDomainError: If ``resistance_1 + resistance_2 == 0``.
    """
    v_in = validate_quantity(
        input_voltage,
        "[electric_potential]",
        name="input_voltage",
    )
    r1 = validate_quantity(
        resistance_1,
        "[resistance]",
        name="resistance_1",
        allow_negative=False,
    )
    r2 = validate_quantity(
        resistance_2,
        "[resistance]",
        name="resistance_2",
        allow_negative=False,
    )
    total_r = r1 + r2
    if float(total_r.magnitude) <= 0:
        raise PhysicalDomainError(
            "Total divider resistance (resistance_1 + resistance_2) must be positive."
        )
    return _to_clean_unit(v_in * (r2 / total_r), "V")


__all__ = [
    "electric_power",
    "joule_heating",
    "ohms_law_current",
    "ohms_law_resistance",
    "ohms_law_voltage",
    "parallel_resistance",
    "series_resistance",
    "voltage_divider",
    "wire_resistance",
]
