"""Hydrostatics formulas: density, hydrostatic pressure, Pascal's principle, and head."""

from __future__ import annotations

import pint

from ..constants import STANDARD_ATMOSPHERE, STANDARD_GRAVITY
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_density, _validate_positive_gravity


def fluid_density(
    mass: QuantityLike,
    volume: QuantityLike,
) -> pint.Quantity:
    """Compute mass density of a fluid: ``rho = m / V``.

    Args:
        mass: Fluid mass (dimension ``[mass]``, ``>= 0``).
        volume: Fluid volume (dimension ``[volume]``, ``> 0``).

    Returns:
        Mass density as a ``pint.Quantity`` in ``kilogram / meter ** 3``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    v = validate_quantity(
        volume,
        "[volume]",
        name="volume",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(m / v, "kg / m^3")


def specific_gravity(
    fluid_density: QuantityLike,
    *,
    reference_density: QuantityLike = "1000 kg / m^3",
) -> pint.Quantity:
    """Compute specific gravity (relative density): ``SG = rho / rho_ref``.

    Args:
        fluid_density: Fluid mass density (dimension ``[density]``, ``>= 0``).
        reference_density: Reference fluid density (dimension ``[density]``, ``> 0``,
            defaults to ``"1000 kg / m^3"`` for water).

    Returns:
        Dimensionless specific gravity as a ``pint.Quantity``.
    """
    rho = _validate_density(fluid_density, name="fluid_density", allow_zero=True)
    rho_ref = _validate_density(
        reference_density,
        name="reference_density",
        allow_zero=False,
    )
    sg = to_si(rho / rho_ref)
    return Quantity(0.0, "dimensionless") if sg.magnitude == 0 else sg


def hydrostatic_gauge_pressure(
    density: QuantityLike,
    depth: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute hydrostatic gauge pressure at depth ``h``: ``P_gauge = rho * g * h``.

    Args:
        density: Incompressible fluid density (dimension ``[density]``, ``>= 0``).
        depth: Vertical depth below the free surface (dimension ``[length]``, ``>= 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Hydrostatic gauge pressure as a ``pint.Quantity`` in ``pascal``.
    """
    rho = _validate_density(density, name="density", allow_zero=True)
    h = validate_quantity(depth, "[length]", name="depth", allow_negative=False)
    g = _validate_positive_gravity(gravity)
    return _to_clean_unit(rho * g * h, "Pa")


def hydrostatic_pressure(
    density: QuantityLike,
    depth: QuantityLike,
    *,
    surface_pressure: QuantityLike = STANDARD_ATMOSPHERE,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute absolute hydrostatic pressure at depth ``h``: ``P = P_0 + rho * g * h``.

    Args:
        density: Incompressible fluid density (dimension ``[density]``, ``>= 0``).
        depth: Vertical depth below the free surface (dimension ``[length]``, ``>= 0``).
        surface_pressure: Pressure at the fluid surface (dimension ``[pressure]``, ``>= 0``,
            defaults to ``STANDARD_ATMOSPHERE``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Absolute hydrostatic pressure as a ``pint.Quantity`` in ``pascal``.
    """
    p0 = validate_quantity(
        surface_pressure,
        "[pressure]",
        name="surface_pressure",
        allow_negative=False,
    )
    p_gauge = hydrostatic_gauge_pressure(density, depth, gravity=gravity)
    return _to_clean_unit(p0 + p_gauge, "Pa")


def pascal_hydraulic_force(
    input_force: QuantityLike,
    input_area: QuantityLike,
    output_area: QuantityLike,
) -> pint.Quantity:
    """Compute hydraulic press output force via Pascal's principle: ``F_2 = F_1 * (A_2 / A_1)``.

    Args:
        input_force: Force applied to the input piston (dimension ``[force]``).
        input_area: Input piston cross-sectional area (dimension ``[area]``, ``> 0``).
        output_area: Output piston cross-sectional area (dimension ``[area]``, ``>= 0``).

    Returns:
        Output force as a ``pint.Quantity`` in ``newton``.
    """
    f1 = validate_quantity(input_force, "[force]", name="input_force")
    a1 = validate_quantity(
        input_area,
        "[area]",
        name="input_area",
        allow_negative=False,
        allow_zero=False,
    )
    a2 = validate_quantity(output_area, "[area]", name="output_area", allow_negative=False)
    return _to_clean_unit(f1 * (a2 / a1), "N")


def pressure_head(
    pressure: QuantityLike,
    density: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute fluid column height (pressure head): ``h = P / (rho * g)``.

    Args:
        pressure: Fluid gauge or static pressure (dimension ``[pressure]``).
        density: Fluid mass density (dimension ``[density]``, ``> 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Pressure head height as a ``pint.Quantity`` in ``meter``.
    """
    p = validate_quantity(pressure, "[pressure]", name="pressure")
    rho = _validate_density(density, name="density", allow_zero=False)
    g = _validate_positive_gravity(gravity)
    return _to_clean_unit(p / (rho * g), "m")


__all__ = [
    "fluid_density",
    "hydrostatic_gauge_pressure",
    "hydrostatic_pressure",
    "pascal_hydraulic_force",
    "pressure_head",
    "specific_gravity",
]
