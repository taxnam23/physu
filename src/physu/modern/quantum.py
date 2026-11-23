"""Basic quantum mechanics formulas: de Broglie wavelength, Heisenberg bounds, and 1D well."""

from __future__ import annotations

import pint

from ..constants import PLANCK_CONSTANT, REDUCED_PLANCK_CONSTANT, SPEED_OF_LIGHT
from ..dynamics.momentum import linear_momentum
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_positive_int
from .photons import _validate_positive_planck
from .relativity import relativistic_momentum


def _heisenberg_bound(
    uncertainty: QuantityLike,
    *,
    dimension: str,
    name: str,
    reduced_planck: QuantityLike,
    target_unit: str,
) -> pint.Quantity:
    """Compute Heisenberg lower bound ``hbar / (2 * uncertainty)`` in ``target_unit``."""
    u = validate_quantity(
        uncertainty,
        dimension,
        name=name,
        allow_negative=False,
        allow_zero=False,
    )
    hbar = validate_quantity(
        reduced_planck,
        "J * s",
        name="reduced_planck",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(hbar / (2.0 * u), target_unit)


def de_broglie_wavelength(
    momentum: QuantityLike,
    *,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute de Broglie matter wavelength from momentum: ``lambda = h / |p|``.

    Args:
        momentum: Non-zero particle momentum (compatible with ``kg * m / s``, ``!= 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).

    Returns:
        de Broglie wavelength as a ``pint.Quantity`` in ``meter``
        (convertible to ``angstrom`` or ``nm``).
    """
    p = validate_quantity(momentum, "kg * m / s", name="momentum", allow_zero=False)
    h = _validate_positive_planck(planck_constant)
    return _to_clean_unit(h / abs(p), "m")


def de_broglie_wavelength_particle(
    mass: QuantityLike,
    speed: QuantityLike,
    *,
    relativistic: bool = False,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute de Broglie wavelength for a massive particle: ``lambda = h / |p|``.

    Uses classical momentum ``p = m * v`` when ``relativistic=False``, or
    relativistic momentum ``p = gamma * m * v`` when ``relativistic=True``.

    Args:
        mass: Positive particle mass (dimension ``[mass]``, ``> 0``).
        speed: Non-zero particle speed (dimension ``[velocity]``, ``!= 0``).
        relativistic: Whether to use relativistic momentum (defaults to ``False``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum used when ``relativistic=True``
            (``[velocity]``, ``> 0``, defaults to ``SPEED_OF_LIGHT``).

    Returns:
        de Broglie wavelength as a ``pint.Quantity`` in ``meter``.
    """
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    v = validate_quantity(speed, "[velocity]", name="speed", allow_zero=False)
    momentum = (
        relativistic_momentum(m, v, speed_of_light=speed_of_light)
        if relativistic
        else linear_momentum(m, v)
    )
    return de_broglie_wavelength(momentum, planck_constant=planck_constant)


def heisenberg_min_momentum_uncertainty(
    position_uncertainty: QuantityLike,
    *,
    reduced_planck: QuantityLike = REDUCED_PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute Heisenberg minimum momentum uncertainty: ``dp_min = hbar / (2 * dx)``.

    Args:
        position_uncertainty: Positive position uncertainty ``dx`` (``[length]``, ``> 0``).
        reduced_planck: Reduced Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``REDUCED_PLANCK_CONSTANT``).

    Returns:
        Minimum momentum uncertainty as a ``pint.Quantity`` in ``kilogram * meter / second``.
    """
    return _heisenberg_bound(
        position_uncertainty,
        dimension="[length]",
        name="position_uncertainty",
        reduced_planck=reduced_planck,
        target_unit="kg * m / s",
    )


def heisenberg_min_energy_uncertainty(
    time_uncertainty: QuantityLike,
    *,
    reduced_planck: QuantityLike = REDUCED_PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute Heisenberg minimum energy uncertainty: ``dE_min = hbar / (2 * dt)``.

    Args:
        time_uncertainty: Positive lifetime or time uncertainty ``dt``
            (dimension ``[time]``, ``> 0``).
        reduced_planck: Reduced Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``REDUCED_PLANCK_CONSTANT``).

    Returns:
        Minimum energy uncertainty as a ``pint.Quantity`` in ``joule``
        (convertible to ``eV``).
    """
    return _heisenberg_bound(
        time_uncertainty,
        dimension="[time]",
        name="time_uncertainty",
        reduced_planck=reduced_planck,
        target_unit="J",
    )


def infinite_well_energy(
    quantum_number: QuantityLike,
    mass: QuantityLike,
    width: QuantityLike,
    *,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute 1D infinite potential well energy eigenvalue: ``E_n = n^2 * h^2 / (8 * m * L^2)``.

    Args:
        quantum_number: Positive integer quantum state ``n >= 1``.
        mass: Confined particle mass (dimension ``[mass]``, ``> 0``).
        width: Potential well width ``L`` (dimension ``[length]``, ``> 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).

    Returns:
        Quantized energy level as a ``pint.Quantity`` in ``joule``
        (convertible to ``eV``).
    """
    n = _validate_positive_int(quantum_number, name="quantum_number")
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    length_q = validate_quantity(
        width,
        "[length]",
        name="width",
        allow_negative=False,
        allow_zero=False,
    )
    h = _validate_positive_planck(planck_constant)
    return _to_clean_unit(((n**2) * (h**2)) / (8.0 * m * (length_q**2)), "J")


__all__ = [
    "de_broglie_wavelength",
    "de_broglie_wavelength_particle",
    "heisenberg_min_energy_uncertainty",
    "heisenberg_min_momentum_uncertainty",
    "infinite_well_energy",
]
