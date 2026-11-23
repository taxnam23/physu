"""Photon, photoelectric effect, Compton scattering, and Wien's displacement formulas."""

from __future__ import annotations

import math

import pint

from ..constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    WIEN_DISPLACEMENT_CONSTANT,
)
from ..exceptions import PhysicalDomainError
from ..thermodynamics.temperature import _to_absolute_kelvin
from ..units import QuantityLike, to_si
from ..validation import validate_quantity
from ..waves.mechanical import wave_frequency
from ._helpers import _to_clean_unit, _validate_positive_c

_TRIG_ZERO_TOL = 1e-15


def _validate_positive_planck(planck_constant: QuantityLike) -> pint.Quantity:
    """Validate strictly positive Planck constant compatible with ``J * s``."""
    return validate_quantity(
        planck_constant,
        "J * s",
        name="planck_constant",
        allow_negative=False,
        allow_zero=False,
    )


def photon_energy(
    *,
    frequency: QuantityLike | None = None,
    wavelength: QuantityLike | None = None,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute photon energy from frequency or vacuum wavelength: ``E = h * f = h * c / lambda``.

    Provide exactly one of ``frequency`` or ``wavelength``.

    Args:
        frequency: Photon frequency (dimension ``[frequency]``, ``>= 0``).
        wavelength: Vacuum photon wavelength (dimension ``[length]``, ``> 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Photon energy as a ``pint.Quantity`` in ``joule`` (convertible to ``eV``).

    Raises:
        ValueError: If neither or both of ``frequency`` and ``wavelength`` are given.
    """
    if (frequency is None) == (wavelength is None):
        raise ValueError("Provide exactly one of 'frequency' or 'wavelength'.")

    h = _validate_positive_planck(planck_constant)
    if frequency is not None:
        freq = validate_quantity(
            frequency,
            "[frequency]",
            name="frequency",
            allow_negative=False,
        )
        return _to_clean_unit(h * freq, "J")

    assert wavelength is not None
    c = _validate_positive_c(speed_of_light)
    lam = validate_quantity(
        wavelength,
        "[length]",
        name="wavelength",
        allow_negative=False,
        allow_zero=False,
    )
    freq_from_lam = wave_frequency(c, lam)
    return _to_clean_unit(h * freq_from_lam, "J")


def photon_momentum(
    *,
    frequency: QuantityLike | None = None,
    wavelength: QuantityLike | None = None,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute photon momentum: ``p = h / lambda = E / c``.

    Provide exactly one of ``frequency`` or ``wavelength``.

    Args:
        frequency: Photon frequency (dimension ``[frequency]``, ``>= 0``).
        wavelength: Vacuum photon wavelength (dimension ``[length]``, ``> 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Photon momentum magnitude as a ``pint.Quantity`` in ``kilogram * meter / second``.
    """
    c = _validate_positive_c(speed_of_light)
    energy = photon_energy(
        frequency=frequency,
        wavelength=wavelength,
        planck_constant=planck_constant,
        speed_of_light=c,
    )
    return _to_clean_unit(energy / c, "kg * m / s")


def photoelectric_max_kinetic_energy(
    frequency: QuantityLike,
    work_function: QuantityLike,
    *,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute maximum photoelectron kinetic energy: ``K_max = h * f - phi``.

    Args:
        frequency: Incident photon frequency (dimension ``[frequency]``, ``>= 0``).
        work_function: Surface work function ``phi`` (dimension ``[energy]``, ``>= 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).

    Returns:
        Maximum photoelectron kinetic energy as a ``pint.Quantity`` in ``joule``.

    Raises:
        PhysicalDomainError: If ``h * f < work_function`` (below threshold).
    """
    e_photon = photon_energy(frequency=frequency, planck_constant=planck_constant)
    phi = validate_quantity(
        work_function,
        "[energy]",
        name="work_function",
        allow_negative=False,
    )
    e_j = float(to_si(e_photon).magnitude)
    phi_j = float(to_si(phi).magnitude)
    if math.isclose(e_j, phi_j, rel_tol=1e-12, abs_tol=0.0):
        return _to_clean_unit(e_photon * 0.0, "J")
    k_max = _to_clean_unit(e_photon - phi, "J")
    if float(to_si(k_max).magnitude) < 0.0:
        raise PhysicalDomainError(
            "Incident photon energy is below the work function (no photoemission)."
        )
    return k_max


def photoelectric_threshold_frequency(
    work_function: QuantityLike,
    *,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
) -> pint.Quantity:
    """Compute photoelectric cutoff threshold frequency: ``f_0 = phi / h``.

    Args:
        work_function: Surface work function ``phi`` (dimension ``[energy]``, ``>= 0``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).

    Returns:
        Threshold frequency as a ``pint.Quantity`` in ``hertz``.
    """
    phi = validate_quantity(
        work_function,
        "[energy]",
        name="work_function",
        allow_negative=False,
    )
    h = _validate_positive_planck(planck_constant)
    return _to_clean_unit(phi / h, "Hz")


def photoelectric_stopping_potential(
    max_kinetic_energy: QuantityLike,
    *,
    charge: QuantityLike = ELEMENTARY_CHARGE,
) -> pint.Quantity:
    """Compute photoelectric stopping voltage: ``V_s = K_max / |q|``.

    Args:
        max_kinetic_energy: Maximum photoelectron kinetic energy (``[energy]``, ``>= 0``).
        charge: Photoelectron charge magnitude (dimension ``[charge]``, ``!= 0``,
            defaults to ``ELEMENTARY_CHARGE``).

    Returns:
        Stopping potential as a ``pint.Quantity`` in ``volt``.
    """
    k_max = validate_quantity(
        max_kinetic_energy,
        "[energy]",
        name="max_kinetic_energy",
        allow_negative=False,
    )
    q = validate_quantity(charge, "[charge]", name="charge", allow_zero=False)
    return _to_clean_unit(k_max / abs(q), "V")


def compton_wavelength_shift(
    scattering_angle: QuantityLike,
    *,
    particle_mass: QuantityLike = ELECTRON_MASS,
    planck_constant: QuantityLike = PLANCK_CONSTANT,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute Compton shift: ``delta_lambda = (h / (m * c)) * (1 - cos(theta))``.

    Args:
        scattering_angle: Photon scattering angle (compatible with ``"rad"``).
        particle_mass: Target charged particle mass (``[mass]``, ``> 0``,
            defaults to ``ELECTRON_MASS``).
        planck_constant: Planck constant (compatible with ``J * s``, ``> 0``,
            defaults to ``PLANCK_CONSTANT``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Compton wavelength shift as a ``pint.Quantity`` in ``meter``.
    """
    theta_rad = float(
        validate_quantity(scattering_angle, "rad", name="scattering_angle")
        .to("rad")
        .magnitude
    )
    m = validate_quantity(
        particle_mass,
        "[mass]",
        name="particle_mass",
        allow_negative=False,
        allow_zero=False,
    )
    h = _validate_positive_planck(planck_constant)
    c = _validate_positive_c(speed_of_light)
    factor = 1.0 - math.cos(theta_rad)
    if abs(factor) < _TRIG_ZERO_TOL:
        factor = 0.0
    elif abs(factor - 2.0) < _TRIG_ZERO_TOL:
        factor = 2.0
    return _to_clean_unit((h / (m * c)) * factor, "m")


def wien_peak_wavelength(
    temperature: QuantityLike,
    *,
    wien_constant: QuantityLike = WIEN_DISPLACEMENT_CONSTANT,
) -> pint.Quantity:
    """Compute blackbody peak emission wavelength via Wien's law: ``lambda_max = b / T``.

    Args:
        temperature: Blackbody absolute temperature (``> 0 K``).
        wien_constant: Wien displacement constant (compatible with ``m * K``, ``> 0``,
            defaults to ``WIEN_DISPLACEMENT_CONSTANT``).

    Returns:
        Peak emission wavelength as a ``pint.Quantity`` in ``meter``.
    """
    t = _to_absolute_kelvin(temperature, name="temperature", allow_zero=False)
    b = validate_quantity(
        wien_constant,
        "m * K",
        name="wien_constant",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(b / t, "m")


__all__ = [
    "compton_wavelength_shift",
    "photoelectric_max_kinetic_energy",
    "photoelectric_stopping_potential",
    "photoelectric_threshold_frequency",
    "photon_energy",
    "photon_momentum",
    "wien_peak_wavelength",
]
