"""2D projectile motion formulas under uniform gravity (positive y-axis upward)."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike
from ..validation import validate_quantity
from . import uniform, vertical

_TRIG_ZERO_TOL = 1e-15


def _snap_trig(value: float) -> float:
    """Snap near-zero trigonometric artifacts (e.g. cos(90 deg)) to exact zero."""
    return 0.0 if abs(value) < _TRIG_ZERO_TOL else value


def _validate_gravity(gravity: QuantityLike) -> pint.Quantity:
    """Validate gravitational acceleration magnitude (strictly positive)."""
    return validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )


def velocity_components(
    initial_speed: QuantityLike,
    angle: QuantityLike,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Decompose launch speed and angle into horizontal and vertical components.

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (unit compatible with ``"rad"``).

    Returns:
        Tuple ``(vx0, vy0)`` as SI base ``pint.Quantity`` values (``meter / second``).

    Raises:
        pint.DimensionalityError: If ``initial_speed`` or ``angle`` has invalid dimensions.
        PhysicalDomainError: If ``initial_speed < 0`` or any input is non-finite.
    """
    speed_m = float(
        validate_quantity(
            initial_speed,
            "[velocity]",
            name="initial_speed",
            allow_negative=False,
        )
        .to("m / s")
        .magnitude
    )
    angle_rad = float(validate_quantity(angle, "rad", name="angle").to("rad").magnitude)

    vx0 = speed_m * _snap_trig(math.cos(angle_rad))
    vy0 = speed_m * _snap_trig(math.sin(angle_rad))
    return Quantity(vx0, "m / s"), Quantity(vy0, "m / s")


def projectile_flight_time(
    initial_speed: QuantityLike,
    angle: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute total flight time until reaching ground level (``y = 0``).

    Solves ``y0 + vy0 * t - 0.5 * g * t^2 = 0`` for ``t >= 0``.

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (compatible with ``"rad"``).
        initial_height: Launch height above ground (``[length]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Flight time as an SI base ``pint.Quantity`` (``second``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If launching downward from ground level (``y0 == 0`` and
            ``vy0 < 0``), ``initial_speed < 0``, ``initial_height < 0``, or ``gravity <= 0``.
    """
    _, vy0_q = velocity_components(initial_speed, angle)
    y0_m = float(
        validate_quantity(
            initial_height,
            "[length]",
            name="initial_height",
            allow_negative=False,
        )
        .to("m")
        .magnitude
    )
    g_m = float(_validate_gravity(gravity).to("m / s^2").magnitude)
    vy0_m = float(vy0_q.magnitude)

    if y0_m == 0.0 and vy0_m < 0.0:
        raise PhysicalDomainError(
            "Cannot launch downward (vy0 < 0) from ground level (initial_height == 0)."
        )

    discriminant = vy0_m**2 + 2.0 * g_m * y0_m
    t_flight = (vy0_m + math.sqrt(discriminant)) / g_m
    return Quantity(max(0.0, t_flight), "s")


def projectile_max_height(
    initial_speed: QuantityLike,
    angle: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute maximum height reached during projectile motion.

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (compatible with ``"rad"``).
        initial_height: Launch height above ground (``[length]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Maximum height as an SI base ``pint.Quantity`` (``meter``).

    Raises:
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If launching downward from ground level (``y0 == 0`` and
            ``vy0 < 0``) or inputs violate domain rules.
    """
    _, vy0 = velocity_components(initial_speed, angle)
    y0 = validate_quantity(
        initial_height,
        "[length]",
        name="initial_height",
        allow_negative=False,
    )
    g = _validate_gravity(gravity)

    if vy0.magnitude < 0:
        if y0.magnitude == 0:
            raise PhysicalDomainError(
                "Cannot launch downward (vy0 < 0) from ground level (initial_height == 0)."
            )
        return y0.to_base_units()

    return vertical.max_height(vy0, initial_height=y0, gravity=g)


def projectile_range(
    initial_speed: QuantityLike,
    angle: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute horizontal range upon reaching ground level (``y = 0``).

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (compatible with ``"rad"``).
        initial_height: Launch height above ground (``[length]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Horizontal range as an SI base ``pint.Quantity`` (``meter``).
    """
    vx0, _ = velocity_components(initial_speed, angle)
    t_flight = projectile_flight_time(
        initial_speed,
        angle,
        initial_height=initial_height,
        gravity=gravity,
    )
    return uniform.displacement(vx0, t_flight)


def projectile_position(
    initial_speed: QuantityLike,
    angle: QuantityLike,
    time: QuantityLike,
    *,
    initial_height: QuantityLike = "0 m",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Compute 2D position ``(x(t), y(t))`` at elapsed ``time >= 0``.

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (compatible with ``"rad"``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).
        initial_height: Initial vertical position (``[length]``, defaults to ``"0 m"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Tuple ``(x, y)`` as SI base ``pint.Quantity`` values (``meter``).
    """
    vx0, vy0 = velocity_components(initial_speed, angle)
    x = uniform.displacement(vx0, time)
    y = vertical.vertical_position(initial_height, vy0, time, gravity=gravity)
    return x, y


def projectile_velocity(
    initial_speed: QuantityLike,
    angle: QuantityLike,
    time: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Compute 2D velocity ``(vx(t), vy(t))`` at elapsed ``time >= 0``.

    Args:
        initial_speed: Launch speed magnitude (``[velocity]``, must be ``>= 0``).
        angle: Launch angle relative to the horizontal (compatible with ``"rad"``).
        time: Elapsed time (dimension ``[time]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Tuple ``(vx, vy)`` as SI base ``pint.Quantity`` values (``meter / second``).
    """
    vx0, vy0 = velocity_components(initial_speed, angle)
    # Validate time even though vx stays constant.
    validate_quantity(time, "[time]", name="time", allow_negative=False)
    vy = vertical.vertical_velocity(vy0, time, gravity=gravity)
    return vx0, vy


__all__ = [
    "projectile_flight_time",
    "projectile_max_height",
    "projectile_position",
    "projectile_range",
    "projectile_velocity",
    "velocity_components",
]
