"""Uniform Circular Motion (UCM) formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity


def _validate_positive_radius(radius: QuantityLike) -> pint.Quantity:
    """Validate circular radius as a strictly positive length."""
    return validate_quantity(
        radius,
        "[length]",
        name="radius",
        allow_negative=False,
        allow_zero=False,
    )


def tangential_speed(
    angular_velocity: QuantityLike,
    radius: QuantityLike,
) -> pint.Quantity:
    """Compute tangential speed magnitude: ``v = |omega| * r``.

    Args:
        angular_velocity: Angular velocity (unit compatible with ``"rad / s"``, e.g. ``"rpm"``).
        radius: Circular path radius (dimension ``[length]``, must be ``> 0``).

    Returns:
        Non-negative tangential speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    omega_rad_s = abs(
        float(
            validate_quantity(angular_velocity, "rad / s", name="angular_velocity")
            .to("rad / s")
            .magnitude
        )
    )
    r_m = float(_validate_positive_radius(radius).to("m").magnitude)
    return Quantity(omega_rad_s * r_m, "m / s")


def angular_velocity_from_speed(
    tangential_speed: QuantityLike,
    radius: QuantityLike,
) -> pint.Quantity:
    """Compute angular speed magnitude from tangential speed: ``omega = v / r``.

    Args:
        tangential_speed: Tangential speed (dimension ``[velocity]``, must be ``>= 0``).
        radius: Circular path radius (dimension ``[length]``, must be ``> 0``).

    Returns:
        Angular velocity as a ``pint.Quantity`` (``radian / second``).
    """
    v_m_s = float(
        validate_quantity(
            tangential_speed,
            "[velocity]",
            name="tangential_speed",
            allow_negative=False,
        )
        .to("m / s")
        .magnitude
    )
    r_m = float(_validate_positive_radius(radius).to("m").magnitude)
    return Quantity(v_m_s / r_m, "rad / s")


def centripetal_acceleration(
    radius: QuantityLike,
    *,
    tangential_speed: QuantityLike | None = None,
    angular_velocity: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute centripetal acceleration: ``a_c = v^2 / r`` or ``a_c = omega^2 * r``.

    Provide exactly one of ``tangential_speed`` or ``angular_velocity``.

    Args:
        radius: Circular path radius (dimension ``[length]``, must be ``> 0``).
        tangential_speed: Tangential speed (dimension ``[velocity]``).
        angular_velocity: Angular velocity (compatible with ``"rad / s"``).

    Returns:
        Centripetal acceleration as an SI base ``pint.Quantity`` (``meter / second ** 2``).

    Raises:
        ValueError: If neither or both of ``tangential_speed`` and
            ``angular_velocity`` are provided.
        pint.DimensionalityError: If any argument has invalid dimensions.
        PhysicalDomainError: If ``radius <= 0`` or any input is non-finite.
    """
    if (tangential_speed is None) == (angular_velocity is None):
        raise ValueError(
            "Provide exactly one of 'tangential_speed' or 'angular_velocity'."
        )

    r = _validate_positive_radius(radius)
    if tangential_speed is not None:
        v = validate_quantity(tangential_speed, "[velocity]", name="tangential_speed")
        return to_unit(v**2 / r, "m / s^2")

    assert angular_velocity is not None
    omega = validate_quantity(angular_velocity, "rad / s", name="angular_velocity")
    return to_unit(omega**2 * r, "m / s^2")


def period_from_frequency(frequency: QuantityLike) -> pint.Quantity:
    """Compute orbital period from cycle frequency: ``T = 1 / f``."""
    freq_hz = float(
        validate_quantity(
            frequency,
            "[frequency]",
            name="frequency",
            allow_negative=False,
            allow_zero=False,
        )
        .to("Hz")
        .magnitude
    )
    return Quantity(1.0 / freq_hz, "s")


def frequency_from_period(period: QuantityLike) -> pint.Quantity:
    """Compute cycle frequency from period: ``f = 1 / T``."""
    period_s = float(
        validate_quantity(
            period,
            "[time]",
            name="period",
            allow_negative=False,
            allow_zero=False,
        )
        .to("s")
        .magnitude
    )
    return Quantity(1.0 / period_s, "Hz")


def angular_velocity_from_period(period: QuantityLike) -> pint.Quantity:
    """Compute angular velocity from period: ``omega = 2 * pi / T``."""
    period_s = float(
        validate_quantity(
            period,
            "[time]",
            name="period",
            allow_negative=False,
            allow_zero=False,
        )
        .to("s")
        .magnitude
    )
    return Quantity(2.0 * math.pi / period_s, "rad / s")


def angular_velocity_from_frequency(frequency: QuantityLike) -> pint.Quantity:
    """Compute angular velocity from cycle frequency: ``omega = 2 * pi * f``."""
    freq_hz = float(
        validate_quantity(
            frequency,
            "[frequency]",
            name="frequency",
            allow_negative=False,
        )
        .to("Hz")
        .magnitude
    )
    return Quantity(2.0 * math.pi * freq_hz, "rad / s")


def period_from_angular_velocity(angular_velocity: QuantityLike) -> pint.Quantity:
    """Compute period from angular velocity: ``T = 2 * pi / |omega|``."""
    omega_rad_s = abs(
        float(
            validate_quantity(
                angular_velocity,
                "rad / s",
                name="angular_velocity",
                allow_zero=False,
            )
            .to("rad / s")
            .magnitude
        )
    )
    if omega_rad_s == 0.0:
        raise PhysicalDomainError("Parameter 'angular_velocity' cannot be zero.")
    return Quantity(2.0 * math.pi / omega_rad_s, "s")


def frequency_from_angular_velocity(angular_velocity: QuantityLike) -> pint.Quantity:
    """Compute cycle frequency from angular velocity: ``f = |omega| / (2 * pi)``."""
    omega_rad_s = abs(
        float(
            validate_quantity(angular_velocity, "rad / s", name="angular_velocity")
            .to("rad / s")
            .magnitude
        )
    )
    return Quantity(omega_rad_s / (2.0 * math.pi), "Hz")


def angular_displacement(
    angular_velocity: QuantityLike,
    time: QuantityLike,
) -> pint.Quantity:
    """Compute angular displacement in UCM: ``dtheta = omega * t``."""
    omega = validate_quantity(angular_velocity, "rad / s", name="angular_velocity")
    elapsed = validate_quantity(time, "[time]", name="time", allow_negative=False)
    return to_unit(omega * elapsed, "rad")


def arc_length(radius: QuantityLike, angle: QuantityLike) -> pint.Quantity:
    """Compute circular arc length: ``s = r * theta``."""
    r = validate_quantity(radius, "[length]", name="radius", allow_negative=False)
    theta = validate_quantity(angle, "rad", name="angle")
    return to_unit(r * theta, "m")


__all__ = [
    "angular_displacement",
    "angular_velocity_from_frequency",
    "angular_velocity_from_period",
    "angular_velocity_from_speed",
    "arc_length",
    "centripetal_acceleration",
    "frequency_from_angular_velocity",
    "frequency_from_period",
    "period_from_angular_velocity",
    "period_from_frequency",
    "tangential_speed",
]
