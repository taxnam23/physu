"""Mechanical work and work-energy theorem formulas with Pint unit validation."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..dynamics.newton import weight
from ..units import Quantity, QuantityLike, to_unit
from ..validation import validate_quantity

_TRIG_ZERO_TOL = 1e-15


def _snap_trig(value: float) -> float:
    """Snap near-zero trigonometric artifacts (e.g. cos(90 deg)) to exact zero."""
    return 0.0 if abs(value) < _TRIG_ZERO_TOL else value


def mechanical_work(
    force: QuantityLike,
    displacement: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute mechanical work done by a constant force: ``W = F * d * cos(theta)``.

    Args:
        force: Constant force (dimension ``[force]``).
        displacement: Displacement along the path (dimension ``[length]``).
        angle: Angle between force and displacement vectors (compatible with
            ``"rad"``, defaults to ``"0 deg"``).

    Returns:
        Mechanical work as a ``pint.Quantity`` in ``joule``.
    """
    f = validate_quantity(force, "[force]", name="force")
    d = validate_quantity(displacement, "[length]", name="displacement")
    angle_rad = float(validate_quantity(angle, "rad", name="angle").to("rad").magnitude)
    cos_theta = _snap_trig(math.cos(angle_rad))
    result = to_unit(f * d * cos_theta, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def net_work(
    initial_kinetic_energy: QuantityLike,
    final_kinetic_energy: QuantityLike,
) -> pint.Quantity:
    """Compute net work via the work-energy theorem: ``W_net = K_f - K_i``.

    Args:
        initial_kinetic_energy: Initial kinetic energy (``[energy]``, ``>= 0``).
        final_kinetic_energy: Final kinetic energy (``[energy]``, ``>= 0``).

    Returns:
        Net work as a ``pint.Quantity`` in ``joule``.
    """
    k_i = validate_quantity(
        initial_kinetic_energy,
        "[energy]",
        name="initial_kinetic_energy",
        allow_negative=False,
    )
    k_f = validate_quantity(
        final_kinetic_energy,
        "[energy]",
        name="final_kinetic_energy",
        allow_negative=False,
    )
    return to_unit(k_f - k_i, "J")


def gravitational_work(
    mass: QuantityLike,
    height_change: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute work done by uniform gravity: ``W_g = -m * g * delta_h``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        height_change: Vertical displacement ``h_f - h_i`` (dimension ``[length]``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Gravitational work as a ``pint.Quantity`` in ``joule``.
    """
    w = weight(mass, gravity=gravity)
    dh = validate_quantity(height_change, "[length]", name="height_change")
    result = to_unit(-w * dh, "J")
    return Quantity(0.0, "J") if result.magnitude == 0 else result


def elastic_work(
    spring_constant: QuantityLike,
    initial_displacement: QuantityLike,
    final_displacement: QuantityLike,
) -> pint.Quantity:
    """Compute work done by an ideal spring: ``W_s = 0.5 * k * (x_i^2 - x_f^2)``.

    Args:
        spring_constant: Spring stiffness (compatible with ``N / m``, ``>= 0``).
        initial_displacement: Initial spring displacement from equilibrium (``[length]``).
        final_displacement: Final spring displacement from equilibrium (``[length]``).

    Returns:
        Elastic spring work as a ``pint.Quantity`` in ``joule``.
    """
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
    )
    x_i = validate_quantity(initial_displacement, "[length]", name="initial_displacement")
    x_f = validate_quantity(final_displacement, "[length]", name="final_displacement")
    return to_unit(0.5 * k * (x_i**2 - x_f**2), "J")


__all__ = [
    "elastic_work",
    "gravitational_work",
    "mechanical_work",
    "net_work",
]
