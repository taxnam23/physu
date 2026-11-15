"""Uniform Rectilinear Motion (MRU) formulas with Pint unit validation."""

from __future__ import annotations

import pint

from ..exceptions import PhysicalDomainError
from ..units import QuantityLike
from ..validation import validate_quantity


def displacement(velocity: QuantityLike, time: QuantityLike) -> pint.Quantity:
    """Compute displacement for constant-velocity 1D motion: ``dx = v * t``.

    Args:
        velocity: Constant velocity (dimension ``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Displacement as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If ``velocity`` or ``time`` has invalid dimensions.
        PhysicalDomainError: If ``time`` is negative or any input is non-finite.
    """
    vel = validate_quantity(velocity, "[velocity]", name="velocity")
    elapsed = validate_quantity(time, "[time]", name="time", allow_negative=False)
    return (vel * elapsed).to_base_units()


def position(
    initial_position: QuantityLike,
    velocity: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute position at ``time`` for constant-velocity 1D motion: ``x = x0 + v * t``.

    Args:
        initial_position: Initial position (dimension ``[length]``).
        velocity: Constant velocity (dimension ``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Position as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time`` is negative or any input is non-finite.
    """
    x0 = validate_quantity(initial_position, "[length]", name="initial_position")
    delta_x = displacement(velocity, time)
    return (x0 + delta_x).to_base_units()


def average_velocity(displacement: QuantityLike, time: QuantityLike) -> pint.Quantity:
    """Compute constant or average velocity: ``v = dx / t``.

    Args:
        displacement: Displacement (dimension ``[length]``).
        time: Elapsed time (dimension ``[time]``, must be ``> 0``).

    Returns:
        Velocity as an SI base ``pint.Quantity`` (``meter / second``).

    Raises:
        pint.DimensionalityError: If arguments have invalid dimensions.
        PhysicalDomainError: If ``time <= 0`` or any input is non-finite.
    """
    delta_x = validate_quantity(displacement, "[length]", name="displacement")
    elapsed = validate_quantity(
        time,
        "[time]",
        name="time",
        allow_negative=False,
        allow_zero=False,
    )
    return (delta_x / elapsed).to_base_units()


def travel_time(displacement: QuantityLike, velocity: QuantityLike) -> pint.Quantity:
    """Compute elapsed time for constant-velocity 1D motion: ``t = dx / v``.

    Args:
        displacement: Target displacement (dimension ``[length]``).
        velocity: Constant velocity (dimension ``[velocity]``, must be ``!= 0``).

    Returns:
        Elapsed time as an SI base ``pint.Quantity`` (``second``).

    Raises:
        pint.DimensionalityError: If arguments have invalid dimensions.
        PhysicalDomainError: If ``velocity == 0``, resulting time is negative
            (displacement and velocity have opposite signs), or any input is non-finite.
    """
    delta_x = validate_quantity(displacement, "[length]", name="displacement")
    vel = validate_quantity(velocity, "[velocity]", name="velocity", allow_zero=False)
    elapsed = (delta_x / vel).to_base_units()
    if elapsed.magnitude < 0:
        raise PhysicalDomainError(
            "Displacement and velocity have opposite signs, yielding a non-physical "
            f"negative time ({elapsed})."
        )
    return elapsed


__all__ = [
    "average_velocity",
    "displacement",
    "position",
    "travel_time",
]
