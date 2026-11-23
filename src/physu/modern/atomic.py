"""Bohr atomic model and hydrogenic spectral transition formulas."""

from __future__ import annotations

import pint

from ..constants import (
    BOHR_RADIUS,
    PLANCK_CONSTANT,
    RYDBERG_CONSTANT,
    SPEED_OF_LIGHT,
)
from ..exceptions import PhysicalDomainError
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_positive_c, _validate_positive_int
from .photons import _validate_positive_planck


def _validate_positive_rydberg(rydberg_constant: QuantityLike) -> pint.Quantity:
    """Validate strictly positive Rydberg constant compatible with ``1 / m``."""
    return validate_quantity(
        rydberg_constant,
        "1 / m",
        name="rydberg_constant",
        allow_negative=False,
        allow_zero=False,
    )


def bohr_orbit_radius(
    principal_quantum_number: QuantityLike,
    *,
    atomic_number: QuantityLike = 1,
    bohr_radius: QuantityLike = BOHR_RADIUS,
) -> pint.Quantity:
    """Compute Bohr orbital radius for a hydrogenic ion: ``r_n = n^2 * a_0 / Z``.

    Args:
        principal_quantum_number: Positive integer principal quantum number ``n >= 1``.
        atomic_number: Positive integer nuclear charge number ``Z >= 1`` (defaults to ``1``).
        bohr_radius: Ground-state Bohr radius (dimension ``[length]``, ``> 0``,
            defaults to ``BOHR_RADIUS``).

    Returns:
        Orbital radius as a ``pint.Quantity`` in ``meter`` (convertible to ``angstrom``).
    """
    n = _validate_positive_int(principal_quantum_number, name="principal_quantum_number")
    z = _validate_positive_int(atomic_number, name="atomic_number")
    a0 = validate_quantity(
        bohr_radius,
        "[length]",
        name="bohr_radius",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(((n**2) * a0) / z, "m")


def bohr_energy_level(
    principal_quantum_number: QuantityLike,
    *,
    atomic_number: QuantityLike = 1,
    rydberg_constant: QuantityLike = RYDBERG_CONSTANT,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute bound energy level of a hydrogenic atom: ``E_n = -R_inf * h * c * Z^2 / n^2``.

    Args:
        principal_quantum_number: Positive integer principal quantum number ``n >= 1``.
        atomic_number: Positive integer nuclear charge number ``Z >= 1`` (defaults to ``1``).
        rydberg_constant: Rydberg constant (compatible with ``1 / m``, ``> 0``,
            defaults to ``RYDBERG_CONSTANT``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Bound state energy as a ``pint.Quantity`` in ``joule`` (convertible to ``eV``).
    """
    n = _validate_positive_int(principal_quantum_number, name="principal_quantum_number")
    z = _validate_positive_int(atomic_number, name="atomic_number")
    r_inf = _validate_positive_rydberg(rydberg_constant)
    h = _validate_positive_planck(planck_constant)
    c = _validate_positive_c(speed_of_light)
    return _to_clean_unit(-(r_inf * h * c * (z**2)) / (n**2), "J")


def bohr_transition_energy(
    initial_n: QuantityLike,
    final_n: QuantityLike,
    *,
    atomic_number: QuantityLike = 1,
    rydberg_constant: QuantityLike = RYDBERG_CONSTANT,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute electronic transition energy: ``delta_E = E(final_n) - E(initial_n)``.

    Positive values correspond to photon absorption (``final_n > initial_n``);
    negative values correspond to photon emission (``final_n < initial_n``).

    Args:
        initial_n: Initial principal quantum number ``n_i >= 1``.
        final_n: Final principal quantum number ``n_f >= 1``.
        atomic_number: Positive integer nuclear charge number ``Z >= 1`` (defaults to ``1``).
        rydberg_constant: Rydberg constant (compatible with ``1 / m``, ``> 0``,
            defaults to ``RYDBERG_CONSTANT``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Signed transition energy as a ``pint.Quantity`` in ``joule``
        (convertible to ``eV``).
    """
    n_i = _validate_positive_int(initial_n, name="initial_n")
    n_f = _validate_positive_int(final_n, name="final_n")
    e_i = bohr_energy_level(
        n_i,
        atomic_number=atomic_number,
        rydberg_constant=rydberg_constant,
        planck_constant=planck_constant,
        speed_of_light=speed_of_light,
    )
    e_f = bohr_energy_level(
        n_f,
        atomic_number=atomic_number,
        rydberg_constant=rydberg_constant,
        planck_constant=planck_constant,
        speed_of_light=speed_of_light,
    )
    return _to_clean_unit(e_f - e_i, "J")


def rydberg_wavelength(
    initial_n: QuantityLike,
    final_n: QuantityLike,
    *,
    atomic_number: QuantityLike = 1,
    rydberg_constant: QuantityLike = RYDBERG_CONSTANT,
) -> pint.Quantity:
    """Compute spectral line wavelength via Rydberg's formula.

    Uses ``1 / lambda = R_inf * Z^2 * |1 / n_f^2 - 1 / n_i^2|``.

    Args:
        initial_n: Initial principal quantum number ``n_i >= 1``.
        final_n: Final principal quantum number ``n_f >= 1`` (``!= initial_n``).
        atomic_number: Positive integer nuclear charge number ``Z >= 1`` (defaults to ``1``).
        rydberg_constant: Rydberg constant (compatible with ``1 / m``, ``> 0``,
            defaults to ``RYDBERG_CONSTANT``).

    Returns:
        Spectral transition wavelength as a ``pint.Quantity`` in ``meter``
        (convertible to ``nm``).

    Raises:
        PhysicalDomainError: If ``initial_n == final_n``.
    """
    n_i = _validate_positive_int(initial_n, name="initial_n")
    n_f = _validate_positive_int(final_n, name="final_n")
    if n_i == n_f:
        raise PhysicalDomainError(
            "Initial and final quantum numbers cannot be equal (zero transition energy)."
        )
    z = _validate_positive_int(atomic_number, name="atomic_number")
    r_inf = _validate_positive_rydberg(rydberg_constant)
    inv_diff = abs((1.0 / (n_f**2)) - (1.0 / (n_i**2)))
    return _to_clean_unit(1.0 / (r_inf * (z**2) * inv_diff), "m")


__all__ = [
    "bohr_energy_level",
    "bohr_orbit_radius",
    "bohr_transition_energy",
    "rydberg_wavelength",
]
