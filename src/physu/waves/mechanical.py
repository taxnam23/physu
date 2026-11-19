"""Mechanical wave propagation, string wave speed, power, and intensity formulas."""

from __future__ import annotations

import math

import pint

from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit


def wave_speed(
    wavelength: QuantityLike,
    frequency: QuantityLike,
) -> pint.Quantity:
    """Compute wave propagation speed: ``v = lambda * f``.

    Args:
        wavelength: Spatial wavelength (dimension ``[length]``, ``>= 0``).
        frequency: Wave frequency (dimension ``[frequency]``, ``>= 0``).

    Returns:
        Wave speed as a ``pint.Quantity`` in ``meter / second``.
    """
    lam = validate_quantity(wavelength, "[length]", name="wavelength", allow_negative=False)
    freq = validate_quantity(frequency, "[frequency]", name="frequency", allow_negative=False)
    return _to_clean_unit(lam * freq, "m / s")


def wavelength(
    wave_speed: QuantityLike,
    frequency: QuantityLike,
) -> pint.Quantity:
    """Compute wavelength from wave speed and frequency: ``lambda = v / f``.

    Args:
        wave_speed: Wave propagation speed (dimension ``[velocity]``, ``>= 0``).
        frequency: Positive wave frequency (dimension ``[frequency]``, ``> 0``).

    Returns:
        Wavelength as a ``pint.Quantity`` in ``meter``.
    """
    v = validate_quantity(wave_speed, "[velocity]", name="wave_speed", allow_negative=False)
    freq = validate_quantity(
        frequency,
        "[frequency]",
        name="frequency",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(v / freq, "m")


def wave_frequency(
    wave_speed: QuantityLike,
    wavelength: QuantityLike,
) -> pint.Quantity:
    """Compute wave frequency from speed and wavelength: ``f = v / lambda``.

    Args:
        wave_speed: Wave propagation speed (dimension ``[velocity]``, ``>= 0``).
        wavelength: Positive wavelength (dimension ``[length]``, ``> 0``).

    Returns:
        Wave frequency as a ``pint.Quantity`` in ``hertz``.
    """
    v = validate_quantity(wave_speed, "[velocity]", name="wave_speed", allow_negative=False)
    lam = validate_quantity(
        wavelength,
        "[length]",
        name="wavelength",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(v / lam, "Hz")


def wavenumber(wavelength: QuantityLike) -> pint.Quantity:
    """Compute angular wavenumber: ``k = 2 * pi / lambda``.

    Args:
        wavelength: Positive spatial wavelength (dimension ``[length]``, ``> 0``).

    Returns:
        Angular wavenumber as a ``pint.Quantity`` in ``radian / meter``.
    """
    lam_m = float(
        validate_quantity(
            wavelength,
            "[length]",
            name="wavelength",
            allow_negative=False,
            allow_zero=False,
        )
        .to("m")
        .magnitude
    )
    return Quantity((2.0 * math.pi) / lam_m, "rad / m")


def wave_phase_speed(
    angular_frequency: QuantityLike,
    wavenumber: QuantityLike,
) -> pint.Quantity:
    """Compute phase speed from angular frequency and wavenumber: ``v = |omega| / k``.

    Args:
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).
        wavenumber: Positive angular wavenumber (compatible with ``"rad / m"``
            or ``"1 / m"``, ``> 0``).

    Returns:
        Phase speed as a ``pint.Quantity`` in ``meter / second``.
    """
    omega_rad_s = abs(
        float(
            validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
            .to("rad / s")
            .magnitude
        )
    )
    k_rad_m = float(
        validate_quantity(
            wavenumber,
            "rad / m",
            name="wavenumber",
            allow_negative=False,
            allow_zero=False,
        )
        .to("rad / m")
        .magnitude
    )
    return Quantity(omega_rad_s / k_rad_m, "m / s")


def string_wave_speed(
    tension: QuantityLike,
    linear_mass_density: QuantityLike,
) -> pint.Quantity:
    """Compute transverse wave speed on a taut string: ``v = sqrt(T / mu)``.

    Args:
        tension: String tension force (dimension ``[force]``, ``>= 0``).
        linear_mass_density: Linear mass density (compatible with ``kg / m``, ``> 0``).

    Returns:
        Wave speed as a ``pint.Quantity`` in ``meter / second``.
    """
    t = validate_quantity(tension, "[force]", name="tension", allow_negative=False)
    mu = validate_quantity(
        linear_mass_density,
        "kg / m",
        name="linear_mass_density",
        allow_negative=False,
        allow_zero=False,
    )
    speed_sq = float(to_si(t / mu).magnitude)
    return Quantity(math.sqrt(speed_sq), "m / s")


def string_wave_power(
    linear_mass_density: QuantityLike,
    wave_speed: QuantityLike,
    angular_frequency: QuantityLike,
    amplitude: QuantityLike,
) -> pint.Quantity:
    """Compute sinusoidal string wave average power: ``P = 0.5 * mu * v * omega^2 * A^2``.

    Args:
        linear_mass_density: Linear mass density (compatible with ``kg / m``, ``>= 0``).
        wave_speed: Transverse wave speed (dimension ``[velocity]``, ``>= 0``).
        angular_frequency: Wave angular frequency (compatible with ``"rad / s"``).
        amplitude: Transverse wave amplitude (dimension ``[length]``, ``>= 0``).

    Returns:
        Average transmitted power as a ``pint.Quantity`` in ``watt``.
    """
    mu = validate_quantity(
        linear_mass_density,
        "kg / m",
        name="linear_mass_density",
        allow_negative=False,
    )
    v = validate_quantity(wave_speed, "[velocity]", name="wave_speed", allow_negative=False)
    omega_rad_s = float(
        validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
        .to("rad / s")
        .magnitude
    )
    amp = validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
    omega_inv_s = Quantity(omega_rad_s, "1 / s")
    return _to_clean_unit(0.5 * mu * v * (omega_inv_s**2) * (amp**2), "W")


def wave_intensity(
    power: QuantityLike,
    area: QuantityLike,
) -> pint.Quantity:
    """Compute average wave intensity over a surface: ``I = P / A``.

    Args:
        power: Average wave power (dimension ``[power]``, ``>= 0``).
        area: Cross-sectional area perpendicular to propagation (dimension ``[area]``, ``> 0``).

    Returns:
        Wave intensity as a ``pint.Quantity`` in ``watt / meter ** 2``.
    """
    p = validate_quantity(power, "[power]", name="power", allow_negative=False)
    surf_area = validate_quantity(
        area,
        "[area]",
        name="area",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(p / surf_area, "W / m^2")


def spherical_wave_intensity(
    power: QuantityLike,
    distance: QuantityLike,
) -> pint.Quantity:
    """Compute isotropic spherical wave intensity at distance ``r``: ``I = P / (4 * pi * r^2)``.

    Args:
        power: Total source power (dimension ``[power]``, ``>= 0``).
        distance: Radial distance from the point source (dimension ``[length]``, ``> 0``).

    Returns:
        Wave intensity as a ``pint.Quantity`` in ``watt / meter ** 2``.
    """
    r = validate_quantity(
        distance,
        "[length]",
        name="distance",
        allow_negative=False,
        allow_zero=False,
    )
    return wave_intensity(power, 4.0 * math.pi * (r**2))


__all__ = [
    "spherical_wave_intensity",
    "string_wave_power",
    "string_wave_speed",
    "wave_frequency",
    "wave_intensity",
    "wave_phase_speed",
    "wave_speed",
    "wavelength",
    "wavenumber",
]
