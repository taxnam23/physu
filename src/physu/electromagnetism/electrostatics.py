"""Electrostatics formulas: Coulomb's law, electric fields, potential, and Gauss flux."""

from __future__ import annotations

import pint

from ..constants import COULOMB_CONSTANT, VACUUM_PERMITTIVITY
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _cos_angle, _to_clean_unit, _validate_positive_factor


def _validate_point_charge_inputs(
    charge: QuantityLike,
    distance: QuantityLike,
    relative_permittivity: QuantityLike,
    *,
    charge_name: str = "charge",
) -> tuple[pint.Quantity, pint.Quantity, float]:
    """Validate signed charge, strictly positive distance, and dielectric constant."""
    q = validate_quantity(charge, "[charge]", name=charge_name)
    r = validate_quantity(
        distance,
        "[length]",
        name="distance",
        allow_negative=False,
        allow_zero=False,
    )
    eps_r = _validate_positive_factor(
        relative_permittivity,
        name="relative_permittivity",
    )
    return q, r, eps_r


def electric_field_point_charge(
    charge: QuantityLike,
    distance: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute radial electric field of a point charge: ``E = k_e * q / (eps_r * r^2)``.

    Args:
        charge: Source electric charge (dimension ``[charge]``).
        distance: Radial distance from the charge (dimension ``[length]``, ``> 0``).
        relative_permittivity: Medium relative permittivity (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Signed radial electric field as a ``pint.Quantity`` in ``volt / meter``
        (convertible to ``newton / coulomb``).
    """
    q, r, eps_r = _validate_point_charge_inputs(
        charge,
        distance,
        relative_permittivity,
    )
    return _to_clean_unit(COULOMB_CONSTANT * q / (eps_r * (r**2)), "V / m")


def electric_force(
    charge: QuantityLike,
    electric_field: QuantityLike,
) -> pint.Quantity:
    """Compute electrostatic force on a charge in an electric field: ``F = q * E``.

    Args:
        charge: Test electric charge (dimension ``[charge]``).
        electric_field: Electric field strength (compatible with ``V / m`` or ``N / C``).

    Returns:
        Signed electrostatic force as a ``pint.Quantity`` in ``newton``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    e_field = validate_quantity(electric_field, "V / m", name="electric_field")
    return _to_clean_unit(q * e_field, "N")


def coulomb_force(
    charge_1: QuantityLike,
    charge_2: QuantityLike,
    distance: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute Coulomb's law force between two point charges: ``F = k_e * q1 * q2 / (eps_r * r^2)``.

    Positive values indicate repulsion (like charges); negative values indicate
    attraction (opposite charges).

    Args:
        charge_1: First electric charge (dimension ``[charge]``).
        charge_2: Second electric charge (dimension ``[charge]``).
        distance: Center-to-center separation (dimension ``[length]``, ``> 0``).
        relative_permittivity: Medium relative permittivity (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Signed electrostatic force as a ``pint.Quantity`` in ``newton``.
    """
    field = electric_field_point_charge(
        charge_1,
        distance,
        relative_permittivity=relative_permittivity,
    )
    return electric_force(charge_2, field)


def uniform_electric_field(
    potential_difference: QuantityLike,
    distance: QuantityLike,
) -> pint.Quantity:
    """Compute uniform electric field between parallel plates: ``E = delta_V / d``.

    Args:
        potential_difference: Electric potential difference (dimension
            ``[electric_potential]``).
        distance: Plate separation (dimension ``[length]``, ``> 0``).

    Returns:
        Electric field as a ``pint.Quantity`` in ``volt / meter``.
    """
    dv = validate_quantity(
        potential_difference,
        "[electric_potential]",
        name="potential_difference",
    )
    d = validate_quantity(
        distance,
        "[length]",
        name="distance",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(dv / d, "V / m")


def electric_potential_point_charge(
    charge: QuantityLike,
    distance: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute electric potential due to a point charge: ``V = k_e * q / (eps_r * r)``.

    Args:
        charge: Source electric charge (dimension ``[charge]``).
        distance: Radial distance from the charge (dimension ``[length]``, ``> 0``).
        relative_permittivity: Medium relative permittivity (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Electric potential as a ``pint.Quantity`` in ``volt``.
    """
    q, r, eps_r = _validate_point_charge_inputs(
        charge,
        distance,
        relative_permittivity,
    )
    return _to_clean_unit(COULOMB_CONSTANT * q / (eps_r * r), "V")


def electric_potential_energy(
    charge_1: QuantityLike,
    charge_2: QuantityLike,
    distance: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute electrostatic potential energy of two charges: ``U_e = k_e * q1 * q2 / (eps_r * r)``.

    Args:
        charge_1: First electric charge (dimension ``[charge]``).
        charge_2: Second electric charge (dimension ``[charge]``).
        distance: Separation distance (dimension ``[length]``, ``> 0``).
        relative_permittivity: Medium relative permittivity (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Electrostatic potential energy as a ``pint.Quantity`` in ``joule``.
    """
    potential = electric_potential_point_charge(
        charge_1,
        distance,
        relative_permittivity=relative_permittivity,
    )
    q2 = validate_quantity(charge_2, "[charge]", name="charge_2")
    return _to_clean_unit(q2 * potential, "J")


def electric_work(
    charge: QuantityLike,
    potential_difference: QuantityLike,
) -> pint.Quantity:
    """Compute work done by an electric field: ``W_e = -q * delta_V``.

    Uses the standard conservative-field sign convention where ``delta_V = V_f - V_i``
    and positive ``W_e`` is work performed *by* the electric field on ``charge``.

    Args:
        charge: Moving electric charge (dimension ``[charge]``).
        potential_difference: Electric potential difference ``V_f - V_i``
            (dimension ``[electric_potential]``).

    Returns:
        Electric field work as a ``pint.Quantity`` in ``joule``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    dv = validate_quantity(
        potential_difference,
        "[electric_potential]",
        name="potential_difference",
    )
    return _to_clean_unit(-q * dv, "J")


def electric_flux(
    electric_field: QuantityLike,
    area: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute electric flux through a flat surface: ``Phi_E = E * A * cos(theta)``.

    Args:
        electric_field: Uniform electric field (compatible with ``V / m`` or ``N / C``).
        area: Surface area (dimension ``[area]``, ``>= 0``).
        angle: Angle between electric field and surface normal (compatible with
            ``"rad"``, defaults to ``"0 deg"``).

    Returns:
        Electric flux as a ``pint.Quantity`` in ``volt * meter``.
    """
    e_field = validate_quantity(electric_field, "V / m", name="electric_field")
    surf_area = validate_quantity(area, "[area]", name="area", allow_negative=False)
    cos_theta = _cos_angle(angle)
    return _to_clean_unit(e_field * surf_area * cos_theta, "V * m")


def gauss_electric_flux(
    enclosed_charge: QuantityLike,
    *,
    relative_permittivity: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute total electric flux via Gauss's law: ``Phi_E = Q_enc / (eps_0 * eps_r)``.

    Args:
        enclosed_charge: Net charge enclosed by the Gaussian surface
            (dimension ``[charge]``).
        relative_permittivity: Medium relative permittivity (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Total electric flux as a ``pint.Quantity`` in ``volt * meter``.
    """
    q_enc = validate_quantity(enclosed_charge, "[charge]", name="enclosed_charge")
    eps_r = _validate_positive_factor(
        relative_permittivity,
        name="relative_permittivity",
    )
    return _to_clean_unit(q_enc / (VACUUM_PERMITTIVITY * eps_r), "V * m")


__all__ = [
    "coulomb_force",
    "electric_field_point_charge",
    "electric_flux",
    "electric_force",
    "electric_potential_energy",
    "electric_potential_point_charge",
    "electric_work",
    "gauss_electric_flux",
    "uniform_electric_field",
]
