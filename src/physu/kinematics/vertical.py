"""Free fall and vertical throw formulas (positive y-axis pointing upward)."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..units import Quantity, QuantityLike
from ..validation import validate_quantity
from . import accelerated


def _validate_gravity(gravity: QuantityLike) -> pint.Quantity:
    """Validate gravitational acceleration magnitude (strictly positive)."""
    return validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )


def vertical_position(
    initial_height: QuantityLike,
    initial_velocity: QuantityLike,
    time: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute height at ``time`` for vertical motion: ``y = y0 + v0 * t - 0.5 * g * t^2``.

    Args:
        initial_height: Initial vertical position (dimension ``[length]``).
        initial_velocity: Initial vertical velocity; positive is upward (``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Vertical position as an SI base ``pint.Quantity`` (``meter``).
    """
    g = _validate_gravity(gravity)
    return accelerated.position(initial_height, initial_velocity, -g, time)


def vertical_velocity(
    initial_velocity: QuantityLike,
    time: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute vertical velocity at ``time``: ``v = v0 - g * t``.

    Args:
        initial_velocity: Initial vertical velocity; positive is upward (``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Vertical velocity as an SI base ``pint.Quantity`` (``meter / second``).
    """
    g = _validate_gravity(gravity)
    return accelerated.final_velocity(initial_velocity, -g, time)


def free_fall_distance(
    time: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute distance fallen from rest in ``time``: ``h = 0.5 * g * t^2``.

    Args:
        time: Fall duration (dimension ``[time]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Non-negative distance fallen as an SI base ``pint.Quantity`` (``meter``).
    """
    g = _validate_gravity(gravity)
    return accelerated.displacement(Quantity(0.0, "m / s"), g, time)


def free_fall_speed(
    height: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute impact speed after falling ``height`` from rest: ``v = sqrt(2 * g * h)``.

    Args:
        height: Fall height (dimension ``[length]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Non-negative speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    h = validate_quantity(height, "[length]", name="height", allow_negative=False)
    g = _validate_gravity(gravity)
    return accelerated.torricelli_velocity(Quantity(0.0, "m / s"), g, h)


def free_fall_time(
    height: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute time to fall ``height`` from rest: ``t = sqrt(2 * h / g)``.

    Args:
        height: Fall height (dimension ``[length]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Fall time as an SI base ``pint.Quantity`` (``second``).
    """
    h_m = float(
        validate_quantity(height, "[length]", name="height", allow_negative=False)
        .to("m")
        .magnitude
    )
    g_m = float(_validate_gravity(gravity).to("m / s^2").magnitude)
    return Quantity(math.sqrt(2.0 * h_m / g_m), "s")


def time_to_peak(
    initial_velocity: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute time to reach maximum height for an upward throw: ``t_peak = v0 / g``.

    Args:
        initial_velocity: Initial upward velocity (``[velocity]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Time to peak as an SI base ``pint.Quantity`` (``second``).
    """
    v0 = validate_quantity(
        initial_velocity,
        "[velocity]",
        name="initial_velocity",
        allow_negative=False,
    )
    g = _validate_gravity(gravity)
    return (v0 / g).to_base_units()


def max_height(
    initial_velocity: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute maximum height reached in an upward throw: ``y_max = y0 + v0^2 / (2 * g)``.

    Args:
        initial_velocity: Initial upward velocity (``[velocity]``, must be ``>= 0``).
        initial_height: Launch height (dimension ``[length]``, defaults to ``"0 m"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Maximum height as an SI base ``pint.Quantity`` (``meter``).
    """
    y0 = validate_quantity(initial_height, "[length]", name="initial_height")
    v0 = validate_quantity(
        initial_velocity,
        "[velocity]",
        name="initial_velocity",
        allow_negative=False,
    )
    g = _validate_gravity(gravity)
    return (y0 + v0**2 / (2.0 * g)).to_base_units()


def time_to_height(
    initial_velocity: QuantityLike,
    target_height: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> tuple[pint.Quantity, ...]:
    """Solve ``y0 + v0 * t - 0.5 * g * t^2 = target_height`` for ``t >= 0``.

    Args:
        initial_velocity: Initial vertical velocity; positive is upward (``[velocity]``).
        target_height: Target vertical position (dimension ``[length]``).
        initial_height: Initial vertical position (``[length]``, defaults to ``"0 m"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        A tuple of 1 or 2 sorted non-negative ``pint.Quantity`` times (``second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``target_height`` is above peak height or only reached
            at negative times.
    """
    y0 = validate_quantity(initial_height, "[length]", name="initial_height")
    y_target = validate_quantity(target_height, "[length]", name="target_height")
    g = _validate_gravity(gravity)
    return accelerated.time_to_displacement(initial_velocity, -g, y_target - y0)


__all__ = [
    "free_fall_distance",
    "free_fall_speed",
    "free_fall_time",
    "max_height",
    "time_to_height",
    "time_to_peak",
    "vertical_position",
    "vertical_velocity",
]
