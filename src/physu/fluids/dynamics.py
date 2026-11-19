"""Fluid dynamics formulas: continuity equation, flow rates, Bernoulli, and Torricelli."""

from __future__ import annotations

import pint

from ..constants import STANDARD_GRAVITY
from ..kinematics.vertical import free_fall_speed
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_density, _validate_positive_gravity


def volumetric_flow_rate(
    area: QuantityLike,
    speed: QuantityLike,
) -> pint.Quantity:
    """Compute volumetric flow rate: ``Q = A * v``.

    Args:
        area: Pipe or channel cross-sectional area (dimension ``[area]``, ``>= 0``).
        speed: Average fluid flow speed (dimension ``[velocity]``, ``>= 0``).

    Returns:
        Volumetric flow rate as a ``pint.Quantity`` in ``meter ** 3 / second``.
    """
    area_q = validate_quantity(area, "[area]", name="area", allow_negative=False)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    return _to_clean_unit(area_q * v, "m^3 / s")


def flow_speed(
    volumetric_flow_rate: QuantityLike,
    area: QuantityLike,
) -> pint.Quantity:
    """Compute average flow speed from volumetric flow rate: ``v = Q / A``.

    Args:
        volumetric_flow_rate: Volumetric flow rate (dimension
            ``[volumetric_flow_rate]``, ``>= 0``).
        area: Cross-sectional area (dimension ``[area]``, ``> 0``).

    Returns:
        Average flow speed as a ``pint.Quantity`` in ``meter / second``.
    """
    q = validate_quantity(
        volumetric_flow_rate,
        "[volumetric_flow_rate]",
        name="volumetric_flow_rate",
        allow_negative=False,
    )
    area_q = validate_quantity(
        area,
        "[area]",
        name="area",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(q / area_q, "m / s")


def mass_flow_rate(
    density: QuantityLike,
    volumetric_flow_rate: QuantityLike,
) -> pint.Quantity:
    """Compute mass flow rate: ``m_dot = rho * Q``.

    Args:
        density: Fluid mass density (dimension ``[density]``, ``>= 0``).
        volumetric_flow_rate: Volumetric flow rate (dimension
            ``[volumetric_flow_rate]``, ``>= 0``).

    Returns:
        Mass flow rate as a ``pint.Quantity`` in ``kilogram / second``.
    """
    rho = _validate_density(density, name="density", allow_zero=True)
    q = validate_quantity(
        volumetric_flow_rate,
        "[volumetric_flow_rate]",
        name="volumetric_flow_rate",
        allow_negative=False,
    )
    return _to_clean_unit(rho * q, "kg / s")


def continuity_speed(
    area_1: QuantityLike,
    speed_1: QuantityLike,
    area_2: QuantityLike,
) -> pint.Quantity:
    """Compute downstream speed for incompressible steady flow: ``v_2 = v_1 * (A_1 / A_2)``.

    Args:
        area_1: Upstream cross-sectional area (dimension ``[area]``, ``>= 0``).
        speed_1: Upstream flow speed (dimension ``[velocity]``, ``>= 0``).
        area_2: Downstream cross-sectional area (dimension ``[area]``, ``> 0``).

    Returns:
        Downstream flow speed as a ``pint.Quantity`` in ``meter / second``.
    """
    a1 = validate_quantity(area_1, "[area]", name="area_1", allow_negative=False)
    v1 = validate_quantity(speed_1, "[velocity]", name="speed_1", allow_negative=False)
    a2 = validate_quantity(
        area_2,
        "[area]",
        name="area_2",
        allow_negative=False,
        allow_zero=False,
    )
    return flow_speed(volumetric_flow_rate(a1, v1), a2)


def dynamic_pressure(
    density: QuantityLike,
    speed: QuantityLike,
) -> pint.Quantity:
    """Compute fluid dynamic pressure: ``q = 0.5 * rho * v^2``.

    Args:
        density: Fluid mass density (dimension ``[density]``, ``>= 0``).
        speed: Flow speed (dimension ``[velocity]``, ``>= 0``).

    Returns:
        Dynamic pressure as a ``pint.Quantity`` in ``pascal``.
    """
    rho = _validate_density(density, name="density", allow_zero=True)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    return _to_clean_unit(0.5 * rho * (v**2), "Pa")


def bernoulli_pressure(
    pressure_1: QuantityLike,
    density: QuantityLike,
    speed_1: QuantityLike,
    height_1: QuantityLike,
    speed_2: QuantityLike,
    height_2: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute downstream static pressure via Bernoulli's equation for incompressible flow.

    Uses ``P_2 = P_1 + 0.5 * rho * (v_1^2 - v_2^2) + rho * g * (h_1 - h_2)``.
    Supports both absolute and gauge pressure inputs.

    Args:
        pressure_1: Upstream static pressure (dimension ``[pressure]``).
        density: Incompressible fluid density (dimension ``[density]``, ``>= 0``).
        speed_1: Upstream flow speed (dimension ``[velocity]``, ``>= 0``).
        height_1: Upstream elevation (dimension ``[length]``).
        speed_2: Downstream flow speed (dimension ``[velocity]``, ``>= 0``).
        height_2: Downstream elevation (dimension ``[length]``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Downstream static pressure as a ``pint.Quantity`` in ``pascal``.
    """
    p1 = validate_quantity(pressure_1, "[pressure]", name="pressure_1")
    rho = _validate_density(density, name="density", allow_zero=True)
    v1 = validate_quantity(speed_1, "[velocity]", name="speed_1", allow_negative=False)
    v2 = validate_quantity(speed_2, "[velocity]", name="speed_2", allow_negative=False)
    h1 = validate_quantity(height_1, "[length]", name="height_1")
    h2 = validate_quantity(height_2, "[length]", name="height_2")
    g = _validate_positive_gravity(gravity)
    q1 = dynamic_pressure(rho, v1)
    q2 = dynamic_pressure(rho, v2)
    return _to_clean_unit(p1 + (q1 - q2) + rho * g * (h1 - h2), "Pa")


def torricelli_efflux_speed(
    depth: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute ideal efflux speed from an orifice under depth ``h``: ``v = sqrt(2 * g * h)``.

    Args:
        depth: Free-surface height above the orifice (dimension ``[length]``, ``>= 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Efflux speed as a ``pint.Quantity`` in ``meter / second``.
    """
    h = validate_quantity(depth, "[length]", name="depth", allow_negative=False)
    g = _validate_positive_gravity(gravity)
    return free_fall_speed(h, gravity=g)


__all__ = [
    "bernoulli_pressure",
    "continuity_speed",
    "dynamic_pressure",
    "flow_speed",
    "mass_flow_rate",
    "torricelli_efflux_speed",
    "volumetric_flow_rate",
]
