"""Static and kinetic friction formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity
from . import newton


def _friction_force(
    coefficient: QuantityLike,
    normal_force: QuantityLike,
    *,
    coefficient_name: str,
) -> pint.Quantity:
    """Compute friction magnitude ``f = mu * N`` with dimensional and sign checks."""
    mu = validate_quantity(
        coefficient,
        "dimensionless",
        name=coefficient_name,
        allow_negative=False,
    )
    normal = validate_quantity(
        normal_force,
        "[force]",
        name="normal_force",
        allow_negative=False,
    )
    return to_unit(mu * normal, "N")


def max_static_friction(
    static_coefficient: QuantityLike,
    normal_force: QuantityLike,
) -> pint.Quantity:
    """Compute maximum static friction force: ``f_s_max = mu_s * N``.

    Args:
        static_coefficient: Coefficient of static friction (dimensionless, ``>= 0``).
        normal_force: Normal contact force magnitude (``[force]``, ``>= 0``).

    Returns:
        Maximum static friction magnitude as a ``pint.Quantity`` in ``newton``.
    """
    return _friction_force(
        static_coefficient,
        normal_force,
        coefficient_name="static_coefficient",
    )


def kinetic_friction(
    kinetic_coefficient: QuantityLike,
    normal_force: QuantityLike,
) -> pint.Quantity:
    """Compute kinetic (sliding) friction force: ``f_k = mu_k * N``.

    Args:
        kinetic_coefficient: Coefficient of kinetic friction (dimensionless, ``>= 0``).
        normal_force: Normal contact force magnitude (``[force]``, ``>= 0``).

    Returns:
        Kinetic friction magnitude as a ``pint.Quantity`` in ``newton``.
    """
    return _friction_force(
        kinetic_coefficient,
        normal_force,
        coefficient_name="kinetic_coefficient",
    )


def friction_coefficient(
    friction_force: QuantityLike,
    normal_force: QuantityLike,
) -> pint.Quantity:
    """Compute coefficient of friction: ``mu = f / N``.

    Args:
        friction_force: Friction force magnitude (``[force]``, ``>= 0``).
        normal_force: Normal force magnitude (``[force]``, ``> 0``).

    Returns:
        Dimensionless coefficient of friction as a ``pint.Quantity``.
    """
    fric = validate_quantity(
        friction_force,
        "[force]",
        name="friction_force",
        allow_negative=False,
    )
    normal = validate_quantity(
        normal_force,
        "[force]",
        name="normal_force",
        allow_negative=False,
        allow_zero=False,
    )
    return to_si(fric / normal)


def normal_force(
    mass: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute normal force on a flat or inclined surface: ``N = m * g * cos(theta)``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        angle: Surface inclination angle in ``[0, 90 deg]`` (defaults to ``"0 deg"``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Normal force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    return newton.inclined_normal_force(mass, angle, gravity=gravity)


def critical_incline_angle(static_coefficient: QuantityLike) -> pint.Quantity:
    """Compute the angle of repose at which sliding begins: ``theta_c = arctan(mu_s)``.

    Args:
        static_coefficient: Coefficient of static friction (dimensionless, ``>= 0``).

    Returns:
        Critical inclination angle as a ``pint.Quantity`` in ``radian``.
    """
    mu_s = float(
        validate_quantity(
            static_coefficient,
            "dimensionless",
            name="static_coefficient",
            allow_negative=False,
        )
        .to("dimensionless")
        .magnitude
    )
    return Quantity(math.atan(mu_s), "rad")


__all__ = [
    "critical_incline_angle",
    "friction_coefficient",
    "kinetic_friction",
    "max_static_friction",
    "normal_force",
]
