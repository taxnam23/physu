"""Mechanical power formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity

_TRIG_ZERO_TOL = 1e-15


def _cos_angle(angle: QuantityLike, *, require_non_zero: bool = False) -> float:
    """Validate ``angle`` and return ``cos(angle)``, snapping near-zero values."""
    angle_rad = float(validate_quantity(angle, "rad", name="angle").to("rad").magnitude)
    cos_theta = math.cos(angle_rad)
    if abs(cos_theta) < _TRIG_ZERO_TOL:
        if require_non_zero:
            raise PhysicalDomainError(
                "Angle cannot be perpendicular to motion (cos(angle) == 0)."
            )
        return 0.0
    return cos_theta


def average_power(
    work: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute average mechanical power: ``P = W / t``.

    Args:
        work: Mechanical work or energy transferred (dimension ``[energy]``).
        time: Elapsed time interval (dimension ``[time]``, ``> 0``).

    Returns:
        Average power as a ``pint.Quantity`` in ``watt``.
    """
    w = validate_quantity(work, "[energy]", name="work")
    t = validate_quantity(
        time,
        "[time]",
        name="time",
        allow_negative=False,
        allow_zero=False,
    )
    return to_unit(w / t, "W")


def instantaneous_power(
    force: QuantityLike,
    velocity: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute instantaneous mechanical power: ``P = F * v * cos(theta)``.

    Args:
        force: Applied force (dimension ``[force]``).
        velocity: Instantaneous velocity (dimension ``[velocity]``).
        angle: Angle between force and velocity (compatible with ``"rad"``,
            defaults to ``"0 deg"``).

    Returns:
        Instantaneous power as a ``pint.Quantity`` in ``watt``.
    """
    f = validate_quantity(force, "[force]", name="force")
    v = validate_quantity(velocity, "[velocity]", name="velocity")
    cos_theta = _cos_angle(angle)
    result = to_unit(f * v * cos_theta, "W")
    return Quantity(0.0, "W") if result.magnitude == 0 else result


def work_from_power(
    power: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute work or energy transferred at constant power: ``W = P * t``.

    Args:
        power: Constant or average power (dimension ``[power]``).
        time: Elapsed time interval (dimension ``[time]``, ``>= 0``).

    Returns:
        Work as a ``pint.Quantity`` in ``joule``.
    """
    p = validate_quantity(power, "[power]", name="power")
    t = validate_quantity(time, "[time]", name="time", allow_negative=False)
    result = to_unit(p * t, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def force_from_power(
    power: QuantityLike,
    velocity: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute force from mechanical power: ``F = P / (v * cos(theta))``.

    Args:
        power: Instantaneous power (dimension ``[power]``).
        velocity: Non-zero velocity (dimension ``[velocity]``, ``!= 0``).
        angle: Angle between force and velocity (compatible with ``"rad"``,
            defaults to ``"0 deg"``, must not satisfy ``cos(angle) == 0``).

    Returns:
        Force as a ``pint.Quantity`` in ``newton``.
    """
    p = validate_quantity(power, "[power]", name="power")
    v = validate_quantity(velocity, "[velocity]", name="velocity", allow_zero=False)
    cos_theta = _cos_angle(angle, require_non_zero=True)
    result = to_unit(p / (v * cos_theta), "N")
    return Quantity(0.0, "N") if result.magnitude == 0 else result


def velocity_from_power(
    power: QuantityLike,
    force: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute velocity from mechanical power: ``v = P / (F * cos(theta))``.

    Args:
        power: Instantaneous power (dimension ``[power]``).
        force: Non-zero force (dimension ``[force]``, ``!= 0``).
        angle: Angle between force and velocity (compatible with ``"rad"``,
            defaults to ``"0 deg"``, must not satisfy ``cos(angle) == 0``).

    Returns:
        Velocity as an SI base ``pint.Quantity`` (``meter / second``).
    """
    p = validate_quantity(power, "[power]", name="power")
    f = validate_quantity(force, "[force]", name="force", allow_zero=False)
    cos_theta = _cos_angle(angle, require_non_zero=True)
    result = to_si(p / (f * cos_theta))
    return Quantity(0.0, "m / s") if result.magnitude == 0 else result


__all__ = [
    "average_power",
    "force_from_power",
    "instantaneous_power",
    "velocity_from_power",
    "work_from_power",
]