"""Newton's laws of motion, weight, and inclined-plane forces."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import require_range, validate_quantity

_TRIG_ZERO_TOL = 1e-15


def _snap_trig(value: float) -> float:
    """Snap near-zero trigonometric artifacts (e.g. cos(90 deg)) to exact zero."""
    return 0.0 if abs(value) < _TRIG_ZERO_TOL else value


def _validate_incline_inputs(
    mass_val: QuantityLike,
    angle: QuantityLike,
    gravity: QuantityLike,
) -> tuple[float, float]:
    """Return ``(weight_in_newtons, angle_in_radians)`` after validating incline inputs."""
    weight_n = float(weight(mass_val, gravity=gravity).to("N").magnitude)
    angle_q = validate_quantity(angle, "rad", name="angle")
    require_range(
        angle_q,
        min_value=Quantity(0.0, "rad"),
        max_value=Quantity(math.pi / 2.0, "rad"),
        name="angle",
    )
    angle_rad = float(angle_q.to("rad").magnitude)
    return weight_n, angle_rad


def force(mass: QuantityLike, acceleration: QuantityLike) -> pint.Quantity:
    """Compute net force using Newton's second law: ``F = m * a``.

    Args:
        mass: Object mass (dimension ``[mass]``, must be ``>= 0``).
        acceleration: Acceleration (dimension ``[acceleration]``).

    Returns:
        Net force as a ``pint.Quantity`` in ``newton``.

    Raises:
        pint.DimensionalityError: If ``mass`` or ``acceleration`` has invalid dimensions.
        PhysicalDomainError: If ``mass < 0`` or any input is non-finite.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    acc = validate_quantity(acceleration, "[acceleration]", name="acceleration")
    return to_unit(m * acc, "N")


def acceleration(force: QuantityLike, mass: QuantityLike) -> pint.Quantity:
    """Compute acceleration using Newton's second law: ``a = F / m``.

    Args:
        force: Net force (dimension ``[force]``).
        mass: Object mass (dimension ``[mass]``, must be ``> 0``).

    Returns:
        Acceleration as an SI base ``pint.Quantity`` (``meter / second ** 2``).

    Raises:
        pint.DimensionalityError: If ``force`` or ``mass`` has invalid dimensions.
        PhysicalDomainError: If ``mass <= 0`` or any input is non-finite.
    """
    net_force = validate_quantity(force, "[force]", name="force")
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    return to_si(net_force / m)


def mass(force: QuantityLike, acceleration: QuantityLike) -> pint.Quantity:
    """Compute inertial mass from Newton's second law: ``m = F / a``.

    Args:
        force: Net force (dimension ``[force]``).
        acceleration: Acceleration (dimension ``[acceleration]``, must be ``!= 0``).

    Returns:
        Mass as an SI base ``pint.Quantity`` (``kilogram``).

    Raises:
        pint.DimensionalityError: If ``force`` or ``acceleration`` has invalid dimensions.
        PhysicalDomainError: If ``acceleration == 0``, ``force`` and ``acceleration``
            have opposite signs (yielding negative mass), or any input is non-finite.
    """
    net_force = validate_quantity(force, "[force]", name="force")
    acc = validate_quantity(
        acceleration,
        "[acceleration]",
        name="acceleration",
        allow_zero=False,
    )
    computed = to_si(net_force / acc)
    if computed.magnitude < 0:
        raise PhysicalDomainError(
            "Force and acceleration have opposite signs, yielding a non-physical "
            f"negative mass ({computed})."
        )
    return computed


def weight(
    mass: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute gravitational weight magnitude: ``W = m * g``.

    Args:
        mass: Object mass (dimension ``[mass]``, must be ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Weight as a ``pint.Quantity`` in ``newton``.
    """
    g = validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )
    return force(mass, g)


def inclined_parallel_force(
    mass: QuantityLike,
    angle: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute gravity component parallel to an inclined plane: ``F_parallel = m * g * sin(theta)``.

    Args:
        mass: Object mass (dimension ``[mass]``, must be ``>= 0``).
        angle: Plane inclination angle in ``[0, 90 deg]`` (compatible with ``"rad"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Parallel weight component as a ``pint.Quantity`` in ``newton``.
    """
    weight_n, angle_rad = _validate_incline_inputs(mass, angle, gravity)
    return Quantity(weight_n * _snap_trig(math.sin(angle_rad)), "N")


def inclined_normal_force(
    mass: QuantityLike,
    angle: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute normal force magnitude on an inclined plane: ``N = m * g * cos(theta)``.

    Args:
        mass: Object mass (dimension ``[mass]``, must be ``>= 0``).
        angle: Plane inclination angle in ``[0, 90 deg]`` (compatible with ``"rad"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Normal force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    weight_n, angle_rad = _validate_incline_inputs(mass, angle, gravity)
    return Quantity(weight_n * _snap_trig(math.cos(angle_rad)), "N")


__all__ = [
    "acceleration",
    "force",
    "inclined_normal_force",
    "inclined_parallel_force",
    "mass",
    "weight",
]
