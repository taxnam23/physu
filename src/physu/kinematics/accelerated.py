"""Uniformly Accelerated Rectilinear Motion (MRUA / SUVAT) formulas."""

from __future__ import annotations

import math

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity

_ZERO_TOLERANCE = 1e-12


def final_velocity(
    initial_velocity: QuantityLike,
    acceleration: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute final velocity: ``v = v0 + a * t``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Final velocity as an SI base ``pint.Quantity`` (``meter / second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time < 0`` or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    acc = validate_quantity(acceleration, "[acceleration]", name="acceleration")
    elapsed = validate_quantity(time, "[time]", name="time", allow_negative=False)
    return to_si(v0 + acc * elapsed)


def displacement(
    initial_velocity: QuantityLike,
    acceleration: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute displacement under constant acceleration: ``dx = v0 * t + 0.5 * a * t^2``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Displacement as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time < 0`` or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    acc = validate_quantity(acceleration, "[acceleration]", name="acceleration")
    elapsed = validate_quantity(time, "[time]", name="time", allow_negative=False)
    return to_si(v0 * elapsed + 0.5 * acc * elapsed**2)


def position(
    initial_position: QuantityLike,
    initial_velocity: QuantityLike,
    acceleration: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute position under constant acceleration: ``x = x0 + v0 * t + 0.5 * a * t^2``.

    Args:
        initial_position: Initial position (dimension ``[length]``).
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Position as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time < 0`` or any input is non-finite.
    """
    x0 = validate_quantity(initial_position, "[length]", name="initial_position")
    delta_x = displacement(initial_velocity, acceleration, time)
    return to_si(x0 + delta_x)


def displacement_from_velocities(
    initial_velocity: QuantityLike,
    final_velocity: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute displacement from average velocity: ``dx = 0.5 * (v0 + v) * t``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        final_velocity: Final velocity (dimension ``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).

    Returns:
        Displacement as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time < 0`` or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    vf = validate_quantity(final_velocity, "[velocity]", name="final_velocity")
    elapsed = validate_quantity(time, "[time]", name="time", allow_negative=False)
    return to_si(0.5 * (v0 + vf) * elapsed)


def acceleration(
    initial_velocity: QuantityLike,
    final_velocity: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute constant acceleration: ``a = (v - v0) / t``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        final_velocity: Final velocity (dimension ``[velocity]``).
        time: Elapsed time (dimension ``[time]``, must be ``> 0``).

    Returns:
        Acceleration as an SI base ``pint.Quantity`` (``meter / second ** 2``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``time <= 0`` or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    vf = validate_quantity(final_velocity, "[velocity]", name="final_velocity")
    elapsed = validate_quantity(
        time,
        "[time]",
        name="time",
        allow_negative=False,
        allow_zero=False,
    )
    return to_si((vf - v0) / elapsed)


def torricelli_velocity(
    initial_velocity: QuantityLike,
    acceleration: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute speed magnitude via Torricelli's equation: ``|v| = sqrt(v0^2 + 2 * a * dx)``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``).
        displacement: Displacement (dimension ``[length]``).

    Returns:
        Non-negative speed magnitude as an SI base ``pint.Quantity`` (``meter / second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``v0^2 + 2 * a * dx < 0`` or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    acc = validate_quantity(acceleration, "[acceleration]", name="acceleration")
    dx = validate_quantity(displacement, "[length]", name="displacement")

    v_squared = (v0**2 + 2.0 * acc * dx).to_base_units()
    mag = float(v_squared.magnitude)
    if mag < 0.0:
        if math.isclose(mag, 0.0, abs_tol=_ZERO_TOLERANCE):
            mag = 0.0
        else:
            raise PhysicalDomainError(
                "Target displacement is unreachable under the given acceleration: "
                f"v^2 = {v_squared} < 0."
            )
    return Quantity(math.sqrt(mag), "m / s")


def time_to_velocity(
    initial_velocity: QuantityLike,
    final_velocity: QuantityLike,
    acceleration: QuantityLike,
) -> pint.Quantity:
    """Compute elapsed time to reach ``final_velocity``: ``t = (v - v0) / a``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        final_velocity: Target velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``, must be ``!= 0``).

    Returns:
        Elapsed time as an SI base ``pint.Quantity`` (``second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``acceleration == 0``, resulting time is negative,
            or any input is non-finite.
    """
    v0 = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    vf = validate_quantity(final_velocity, "[velocity]", name="final_velocity")
    acc = validate_quantity(
        acceleration,
        "[acceleration]",
        name="acceleration",
        allow_zero=False,
    )
    elapsed = to_si((vf - v0) / acc)
    if elapsed.magnitude < 0:
        raise PhysicalDomainError(
            "Target velocity cannot be reached at non-negative time with the given "
            f"acceleration ({elapsed})."
        )
    return elapsed


def time_to_displacement(
    initial_velocity: QuantityLike,
    acceleration: QuantityLike,
    displacement: QuantityLike,
) -> tuple[pint.Quantity, ...]:
    """Solve ``0.5 * a * t^2 + v0 * t = dx`` for non-negative elapsed time(s) ``t >= 0``.

    Args:
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        acceleration: Constant acceleration (dimension ``[acceleration]``).
        displacement: Target displacement (dimension ``[length]``).

    Returns:
        A tuple of 1 or 2 sorted non-negative ``pint.Quantity`` times (``second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If the displacement is unreachable, indeterminate
            (``v0 == 0``, ``a == 0``, ``dx == 0``), or only occurs for ``t < 0``.
    """
    v0 = float(
        validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
        .to("m / s")
        .magnitude
    )
    acc = float(
        validate_quantity(acceleration, "[acceleration]", name="acceleration")
        .to("m / s^2")
        .magnitude
    )
    dx = float(
        validate_quantity(displacement, "[length]", name="displacement")
        .to("m")
        .magnitude
    )

    if math.isclose(acc, 0.0, abs_tol=_ZERO_TOLERANCE):
        if math.isclose(v0, 0.0, abs_tol=_ZERO_TOLERANCE):
            if math.isclose(dx, 0.0, abs_tol=_ZERO_TOLERANCE):
                raise PhysicalDomainError(
                    "Time is indeterminate when initial_velocity, acceleration, "
                    "and displacement are all zero."
                )
            raise PhysicalDomainError(
                "Target non-zero displacement is unreachable when initial_velocity "
                "and acceleration are both zero."
            )
        t_linear = dx / v0
        if t_linear < -_ZERO_TOLERANCE:
            raise PhysicalDomainError(
                f"Target displacement is only reached at negative time ({t_linear} s)."
            )
        return (Quantity(max(0.0, t_linear), "s"),)

    discriminant = v0**2 + 2.0 * acc * dx
    if discriminant < -_ZERO_TOLERANCE:
        raise PhysicalDomainError(
            "Target displacement is unreachable: quadratic discriminant is negative "
            f"({discriminant} m^2 / s^2)."
        )

    sqrt_disc = math.sqrt(max(0.0, discriminant))
    raw_roots = [(-v0 - sqrt_disc) / acc, (-v0 + sqrt_disc) / acc]

    valid_roots: list[float] = []
    for root in sorted(raw_roots):
        if root >= -_ZERO_TOLERANCE:
            snapped = 0.0 if root < 0.0 else root
            if not any(
                math.isclose(snapped, existing, abs_tol=_ZERO_TOLERANCE)
                for existing in valid_roots
            ):
                valid_roots.append(snapped)

    if not valid_roots:
        raise PhysicalDomainError(
            "Target displacement is only reached at negative (non-physical) times."
        )

    return tuple(Quantity(t_val, "s") for t_val in valid_roots)


__all__ = [
    "acceleration",
    "displacement",
    "displacement_from_velocities",
    "final_velocity",
    "position",
    "time_to_displacement",
    "time_to_velocity",
    "torricelli_velocity",
]
