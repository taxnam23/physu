"""Buoyancy formulas: Archimedes' principle, apparent weight, and floating equilibrium."""

from __future__ import annotations

import pint

from ..constants import STANDARD_GRAVITY
from ..dynamics.newton import weight
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_density, _validate_positive_gravity


def buoyant_force(
    fluid_density: QuantityLike,
    submerged_volume: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute upward buoyant force via Archimedes' principle: ``F_B = rho_f * V_sub * g``.

    Args:
        fluid_density: Surrounding fluid mass density (``[density]``, ``>= 0``).
        submerged_volume: Displaced fluid volume (dimension ``[volume]``, ``>= 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Buoyant force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    rho_f = _validate_density(fluid_density, name="fluid_density", allow_zero=True)
    v_sub = validate_quantity(
        submerged_volume,
        "[volume]",
        name="submerged_volume",
        allow_negative=False,
    )
    g = _validate_positive_gravity(gravity)
    return _to_clean_unit(rho_f * v_sub * g, "N")


def apparent_weight(
    object_mass: QuantityLike,
    object_volume: QuantityLike,
    fluid_density: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute apparent weight of a submerged body: ``W_app = m * g - rho_f * V * g``.

    Positive values indicate net downward weight (body sinks); zero indicates neutral
    buoyancy; negative values indicate net upward buoyant force.

    Args:
        object_mass: Mass of the submerged object (dimension ``[mass]``, ``>= 0``).
        object_volume: Total volume of the object (dimension ``[volume]``, ``>= 0``).
        fluid_density: Surrounding fluid mass density (``[density]``, ``>= 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Signed apparent weight as a ``pint.Quantity`` in ``newton``.
    """
    w_true = weight(object_mass, gravity=gravity)
    v_obj = validate_quantity(
        object_volume,
        "[volume]",
        name="object_volume",
        allow_negative=False,
    )
    f_b = buoyant_force(fluid_density, v_obj, gravity=gravity)
    return _to_clean_unit(w_true - f_b, "N")


def floating_submerged_fraction(
    object_density: QuantityLike,
    fluid_density: QuantityLike,
) -> pint.Quantity:
    """Compute submerged volume fraction for a floating body: ``f_sub = rho_obj / rho_f``.

    Args:
        object_density: Average density of the floating object (``[density]``, ``>= 0``).
        fluid_density: Fluid mass density (``[density]``, ``> 0``, ``>= object_density``).

    Returns:
        Dimensionless submerged volume fraction in ``[0, 1]``.

    Raises:
        PhysicalDomainError: If ``object_density > fluid_density`` (body sinks).
    """
    rho_obj = _validate_density(object_density, name="object_density", allow_zero=True)
    rho_f = _validate_density(fluid_density, name="fluid_density", allow_zero=False)
    if float(to_si(rho_obj).magnitude) > float(to_si(rho_f).magnitude):
        raise PhysicalDomainError(
            "Object density cannot exceed fluid density for floating equilibrium."
        )
    frac = to_si(rho_obj / rho_f)
    return Quantity(0.0, "dimensionless") if frac.magnitude == 0 else frac


def submerged_volume_floating(
    object_mass: QuantityLike,
    fluid_density: QuantityLike,
) -> pint.Quantity:
    """Compute displaced fluid volume of a freely floating body: ``V_sub = m / rho_f``.

    Args:
        object_mass: Mass of the floating body (dimension ``[mass]``, ``>= 0``).
        fluid_density: Fluid mass density (``[density]``, ``> 0``).

    Returns:
        Displaced fluid volume as a ``pint.Quantity`` in ``meter ** 3``.
    """
    m = validate_quantity(object_mass, "[mass]", name="object_mass", allow_negative=False)
    rho_f = _validate_density(fluid_density, name="fluid_density", allow_zero=False)
    return _to_clean_unit(m / rho_f, "m^3")


__all__ = [
    "apparent_weight",
    "buoyant_force",
    "floating_submerged_fraction",
    "submerged_volume_floating",
]
