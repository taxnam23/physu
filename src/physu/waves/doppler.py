"""Acoustic and relativistic Doppler shifts, Mach number, and Mach cone formulas."""

from __future__ import annotations

import math

import pint

from ..constants import SPEED_OF_LIGHT
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit


def doppler_frequency(
    source_frequency: QuantityLike,
    wave_speed: QuantityLike,
    *,
    observer_velocity: QuantityLike = "0 m / s",
    source_velocity: QuantityLike = "0 m / s",
) -> pint.Quantity:
    """Compute classical Doppler-shifted frequency: ``f' = f * (v + v_o) / (v - v_s)``.

    Uses the standard sign convention where positive ``observer_velocity`` and
    positive ``source_velocity`` represent motion *toward* the other party
    (increasing observed frequency), while negative values represent motion
    *away* (decreasing observed frequency).

    Args:
        source_frequency: Emitted wave frequency (dimension ``[frequency]``, ``>= 0``).
        wave_speed: Wave propagation speed in the medium (dimension ``[velocity]``, ``> 0``).
        observer_velocity: Observer speed along the line of sight (dimension
            ``[velocity]``, defaults to ``"0 m / s"``).
        source_velocity: Source speed along the line of sight (dimension
            ``[velocity]``, defaults to ``"0 m / s"``).

    Returns:
        Observed frequency as a ``pint.Quantity`` in ``hertz``.

    Raises:
        PhysicalDomainError: If ``wave_speed + observer_velocity < 0`` or
            ``wave_speed - source_velocity <= 0``.
    """
    f_s = validate_quantity(
        source_frequency,
        "[frequency]",
        name="source_frequency",
        allow_negative=False,
    )
    v = validate_quantity(
        wave_speed,
        "[velocity]",
        name="wave_speed",
        allow_negative=False,
        allow_zero=False,
    )
    v_o = validate_quantity(observer_velocity, "[velocity]", name="observer_velocity")
    v_s = validate_quantity(source_velocity, "[velocity]", name="source_velocity")

    num = v + v_o
    den = v - v_s
    if float(to_si(num).magnitude) < 0:
        raise PhysicalDomainError(
            "Observer receding faster than wave speed yields non-physical negative frequency."
        )
    if float(to_si(den).magnitude) <= 0:
        raise PhysicalDomainError(
            "Source speed toward observer must be strictly less than wave speed."
        )
    return _to_clean_unit(f_s * (num / den), "Hz")


def relativistic_doppler_frequency(
    source_frequency: QuantityLike,
    relative_velocity: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute relativistic Doppler frequency: ``f' = f * sqrt((1 + beta) / (1 - beta))``.

    Positive ``relative_velocity`` corresponds to source and observer approaching
    each other (blueshift, ``f' > f``); negative ``relative_velocity`` corresponds
    to recession (redshift, ``f' < f``).

    Args:
        source_frequency: Proper emitted frequency (dimension ``[frequency]``, ``>= 0``).
        relative_velocity: Line-of-sight approach velocity (``[velocity]``,
            satisfying ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (dimension ``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Observed frequency as a ``pint.Quantity`` in ``hertz``.

    Raises:
        PhysicalDomainError: If ``|relative_velocity| >= speed_of_light``.
    """
    f_s = validate_quantity(
        source_frequency,
        "[frequency]",
        name="source_frequency",
        allow_negative=False,
    )
    v_rel = validate_quantity(relative_velocity, "[velocity]", name="relative_velocity")
    c = validate_quantity(
        speed_of_light,
        "[velocity]",
        name="speed_of_light",
        allow_negative=False,
        allow_zero=False,
    )
    beta = float(to_si(v_rel / c).magnitude)
    if abs(beta) >= 1.0:
        raise PhysicalDomainError(
            "Relative velocity magnitude must be strictly less than the speed of light."
        )
    factor = math.sqrt((1.0 + beta) / (1.0 - beta))
    return _to_clean_unit(f_s * factor, "Hz")


def mach_number(
    object_speed: QuantityLike,
    sound_speed: QuantityLike,
) -> pint.Quantity:
    """Compute Mach number: ``M = v / c_s``.

    Args:
        object_speed: Object speed relative to the medium (``[velocity]``, ``>= 0``).
        sound_speed: Local speed of sound in the medium (``[velocity]``, ``> 0``).

    Returns:
        Dimensionless Mach number as a ``pint.Quantity``.
    """
    v = validate_quantity(object_speed, "[velocity]", name="object_speed", allow_negative=False)
    c_s = validate_quantity(
        sound_speed,
        "[velocity]",
        name="sound_speed",
        allow_negative=False,
        allow_zero=False,
    )
    return to_si(v / c_s)


def mach_angle(
    object_speed: QuantityLike,
    sound_speed: QuantityLike,
) -> pint.Quantity:
    """Compute Mach shock-wave cone half-angle: ``mu = arcsin(c_s / v)``.

    Args:
        object_speed: Sonic or supersonic object speed (``[velocity]``, ``>= sound_speed``).
        sound_speed: Local speed of sound in the medium (``[velocity]``, ``> 0``).

    Returns:
        Mach cone half-angle as a ``pint.Quantity`` in ``radian``
        (convertible to ``degree``).

    Raises:
        PhysicalDomainError: If ``object_speed < sound_speed`` (subsonic ``M < 1``).
    """
    mach = mach_number(object_speed, sound_speed)
    m_val = float(mach.magnitude)
    if m_val < 1.0:
        raise PhysicalDomainError(
            f"Mach angle is only defined for sonic or supersonic speed (Mach >= 1), got {m_val}."
        )
    return Quantity(math.asin(min(1.0, 1.0 / m_val)), "rad")


__all__ = [
    "doppler_frequency",
    "mach_angle",
    "mach_number",
    "relativistic_doppler_frequency",
]
