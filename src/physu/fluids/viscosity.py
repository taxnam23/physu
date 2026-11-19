"""Viscosity formulas: kinematic viscosity, Reynolds number, Poiseuille flow, and Stokes drag."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import (
    _to_clean_unit,
    _validate_density,
    _validate_positive_gravity,
    _validate_viscosity,
)


def kinematic_viscosity(
    dynamic_viscosity: QuantityLike,
    density: QuantityLike,
) -> pint.Quantity:
    """Compute kinematic viscosity: ``nu = mu / rho``.

    Args:
        dynamic_viscosity: Fluid dynamic viscosity (``[viscosity]``, ``>= 0``).
        density: Fluid mass density (``[density]``, ``> 0``).

    Returns:
        Kinematic viscosity as a ``pint.Quantity`` in ``meter ** 2 / second``
        (convertible to ``cSt`` or ``stokes``).
    """
    mu = _validate_viscosity(dynamic_viscosity, name="dynamic_viscosity", allow_zero=True)
    rho = _validate_density(density, name="density", allow_zero=False)
    return _to_clean_unit(mu / rho, "m^2 / s")


def reynolds_number(
    density: QuantityLike,
    speed: QuantityLike,
    characteristic_length: QuantityLike,
    dynamic_viscosity: QuantityLike,
) -> pint.Quantity:
    """Compute Reynolds number: ``Re = rho * v * L / mu``.

    Args:
        density: Fluid mass density (``[density]``, ``>= 0``).
        speed: Characteristic flow speed (``[velocity]``, ``>= 0``).
        characteristic_length: Characteristic linear dimension (``[length]``, ``>= 0``).
        dynamic_viscosity: Positive dynamic viscosity (``[viscosity]``, ``> 0``).

    Returns:
        Dimensionless Reynolds number as a ``pint.Quantity``.
    """
    rho = _validate_density(density, name="density", allow_zero=True)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    length_q = validate_quantity(
        characteristic_length,
        "[length]",
        name="characteristic_length",
        allow_negative=False,
    )
    mu = _validate_viscosity(dynamic_viscosity, name="dynamic_viscosity", allow_zero=False)
    re = to_si((rho * v * length_q) / mu)
    return Quantity(0.0, "dimensionless") if re.magnitude == 0 else re


def poiseuille_flow_rate(
    pressure_drop: QuantityLike,
    radius: QuantityLike,
    dynamic_viscosity: QuantityLike,
    length: QuantityLike,
) -> pint.Quantity:
    """Compute Hagen-Poiseuille laminar pipe flow rate: ``Q = pi * dP * r^4 / (8 * mu * L)``.

    Args:
        pressure_drop: Pressure drop along pipe (``[pressure]``, ``>= 0``).
        radius: Inner circular pipe radius (``[length]``, ``>= 0``).
        dynamic_viscosity: Positive dynamic viscosity (``[viscosity]``, ``> 0``).
        length: Pipe segment length (``[length]``, ``> 0``).

    Returns:
        Volumetric flow rate as a ``pint.Quantity`` in ``meter ** 3 / second``.
    """
    dp = validate_quantity(pressure_drop, "[pressure]", name="pressure_drop", allow_negative=False)
    r = validate_quantity(radius, "[length]", name="radius", allow_negative=False)
    mu = _validate_viscosity(dynamic_viscosity, name="dynamic_viscosity", allow_zero=False)
    length_q = validate_quantity(
        length,
        "[length]",
        name="length",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit((math.pi * dp * (r**4)) / (8.0 * mu * length_q), "m^3 / s")


def stokes_drag_force(
    dynamic_viscosity: QuantityLike,
    radius: QuantityLike,
    speed: QuantityLike,
) -> pint.Quantity:
    """Compute Stokes drag force on a small sphere: ``F_d = 6 * pi * mu * r * v``.

    Args:
        dynamic_viscosity: Fluid dynamic viscosity (``[viscosity]``, ``>= 0``).
        radius: Sphere radius (``[length]``, ``>= 0``).
        speed: Sphere speed relative to fluid (``[velocity]``, ``>= 0``).

    Returns:
        Viscous drag force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    mu = _validate_viscosity(dynamic_viscosity, name="dynamic_viscosity", allow_zero=True)
    r = validate_quantity(radius, "[length]", name="radius", allow_negative=False)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    return _to_clean_unit(6.0 * math.pi * mu * r * v, "N")


def stokes_terminal_velocity(
    sphere_radius: QuantityLike,
    sphere_density: QuantityLike,
    fluid_density: QuantityLike,
    dynamic_viscosity: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute Stokes terminal velocity: ``v_t = 2 * r^2 * (rho_s - rho_f) * g / (9 * mu)``.

    Positive values indicate downward settling (``rho_s > rho_f``); negative
    values indicate upward rising (e.g., gas bubbles with ``rho_s < rho_f``).

    Args:
        sphere_radius: Sphere radius (``[length]``, ``>= 0``).
        sphere_density: Sphere material density (``[density]``, ``>= 0``).
        fluid_density: Surrounding fluid density (``[density]``, ``>= 0``).
        dynamic_viscosity: Positive dynamic viscosity (``[viscosity]``, ``> 0``).
        gravity: Gravitational acceleration (``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Signed terminal velocity as a ``pint.Quantity`` in ``meter / second``.
    """
    r = validate_quantity(sphere_radius, "[length]", name="sphere_radius", allow_negative=False)
    rho_s = _validate_density(sphere_density, name="sphere_density", allow_zero=True)
    rho_f = _validate_density(fluid_density, name="fluid_density", allow_zero=True)
    mu = _validate_viscosity(dynamic_viscosity, name="dynamic_viscosity", allow_zero=False)
    g = _validate_positive_gravity(gravity)
    return _to_clean_unit((2.0 * (r**2) * (rho_s - rho_f) * g) / (9.0 * mu), "m / s")


__all__ = [
    "kinematic_viscosity",
    "poiseuille_flow_rate",
    "reynolds_number",
    "stokes_drag_force",
    "stokes_terminal_velocity",
]
