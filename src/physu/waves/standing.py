"""Standing waves, harmonic frequencies and wavelengths, string tension, and beats."""

from __future__ import annotations

import pint

from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_harmonic
from .mechanical import wave_frequency, wave_speed


def _harmonic_wavelength(
    harmonic: QuantityLike,
    length: QuantityLike,
    *,
    factor: float,
    odd_only: bool,
) -> pint.Quantity:
    """Compute standing-wave harmonic wavelength ``factor * L / n`` in ``meter``."""
    n = _validate_harmonic(harmonic, odd_only=odd_only)
    length_q = validate_quantity(
        length,
        "[length]",
        name="length",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(factor * length_q / n, "m")


def _harmonic_frequency(
    harmonic: QuantityLike,
    length: QuantityLike,
    speed: QuantityLike,
    *,
    factor: float,
    odd_only: bool,
) -> pint.Quantity:
    """Compute standing-wave harmonic frequency ``v / lambda_n`` in ``hertz``."""
    lam_n = _harmonic_wavelength(harmonic, length, factor=factor, odd_only=odd_only)
    return wave_frequency(speed, lam_n)


def string_harmonic_wavelength(
    harmonic: QuantityLike,
    length: QuantityLike,
) -> pint.Quantity:
    """Compute harmonic wavelength on a string fixed at both ends: ``lambda_n = 2 * L / n``.

    Args:
        harmonic: Positive integer harmonic number ``n >= 1``.
        length: String length between fixed nodes (dimension ``[length]``, ``> 0``).

    Returns:
        Harmonic wavelength as a ``pint.Quantity`` in ``meter``.
    """
    return _harmonic_wavelength(harmonic, length, factor=2.0, odd_only=False)


def closed_pipe_harmonic_wavelength(
    harmonic: QuantityLike,
    length: QuantityLike,
) -> pint.Quantity:
    """Compute harmonic wavelength in a pipe closed at one end: ``lambda_n = 4 * L / n``.

    Args:
        harmonic: Positive odd integer harmonic number (``1, 3, 5, ...``).
        length: Pipe length (dimension ``[length]``, ``> 0``).

    Returns:
        Harmonic wavelength as a ``pint.Quantity`` in ``meter``.
    """
    return _harmonic_wavelength(harmonic, length, factor=4.0, odd_only=True)


def string_harmonic_frequency(
    harmonic: QuantityLike,
    length: QuantityLike,
    wave_speed: QuantityLike,
) -> pint.Quantity:
    """Compute harmonic frequency of a fixed string: ``f_n = n * v / (2 * L)``.

    Args:
        harmonic: Positive integer harmonic number ``n >= 1``.
        length: String length between fixed ends (dimension ``[length]``, ``> 0``).
        wave_speed: Wave propagation speed along the string (``[velocity]``, ``>= 0``).

    Returns:
        Harmonic frequency as a ``pint.Quantity`` in ``hertz``.
    """
    return _harmonic_frequency(harmonic, length, wave_speed, factor=2.0, odd_only=False)


def open_pipe_harmonic_frequency(
    harmonic: QuantityLike,
    length: QuantityLike,
    wave_speed: QuantityLike,
) -> pint.Quantity:
    """Compute harmonic frequency of a pipe open at both ends: ``f_n = n * v / (2 * L)``.

    Args:
        harmonic: Positive integer harmonic number ``n >= 1``.
        length: Acoustic pipe length (dimension ``[length]``, ``> 0``).
        wave_speed: Speed of sound in the pipe medium (``[velocity]``, ``>= 0``).

    Returns:
        Harmonic frequency as a ``pint.Quantity`` in ``hertz``.
    """
    return string_harmonic_frequency(harmonic, length, wave_speed)


def closed_pipe_harmonic_frequency(
    harmonic: QuantityLike,
    length: QuantityLike,
    wave_speed: QuantityLike,
) -> pint.Quantity:
    """Compute harmonic frequency of a pipe closed at one end: ``f_n = n * v / (4 * L)``.

    Args:
        harmonic: Positive odd integer harmonic number (``1, 3, 5, ...``).
        length: Acoustic pipe length (dimension ``[length]``, ``> 0``).
        wave_speed: Speed of sound in the pipe medium (``[velocity]``, ``>= 0``).

    Returns:
        Harmonic frequency as a ``pint.Quantity`` in ``hertz``.
    """
    return _harmonic_frequency(harmonic, length, wave_speed, factor=4.0, odd_only=True)


def fundamental_tension_for_frequency(
    frequency: QuantityLike,
    length: QuantityLike,
    linear_mass_density: QuantityLike,
) -> pint.Quantity:
    """Compute string tension required for a fundamental frequency: ``T = mu * (2 * L * f_1)^2``.

    Args:
        frequency: Desired fundamental frequency ``f_1`` (``[frequency]``, ``>= 0``).
        length: Vibrating string length (dimension ``[length]``, ``> 0``).
        linear_mass_density: Linear mass density (compatible with ``kg / m``, ``>= 0``).

    Returns:
        Required string tension as a ``pint.Quantity`` in ``newton``.
    """
    lam_1 = string_harmonic_wavelength(1, length)
    speed = wave_speed(lam_1, frequency)
    mu = validate_quantity(
        linear_mass_density,
        "kg / m",
        name="linear_mass_density",
        allow_negative=False,
    )
    return _to_clean_unit(mu * (speed**2), "N")


def beat_frequency(
    frequency_1: QuantityLike,
    frequency_2: QuantityLike,
) -> pint.Quantity:
    """Compute acoustic beat frequency from two interfering waves: ``f_beat = |f_1 - f_2|``.

    Args:
        frequency_1: First wave frequency (dimension ``[frequency]``, ``>= 0``).
        frequency_2: Second wave frequency (dimension ``[frequency]``, ``>= 0``).

    Returns:
        Beat frequency as a ``pint.Quantity`` in ``hertz``.
    """
    f1 = validate_quantity(frequency_1, "[frequency]", name="frequency_1", allow_negative=False)
    f2 = validate_quantity(frequency_2, "[frequency]", name="frequency_2", allow_negative=False)
    return _to_clean_unit(abs(f1 - f2), "Hz")


__all__ = [
    "beat_frequency",
    "closed_pipe_harmonic_frequency",
    "closed_pipe_harmonic_wavelength",
    "fundamental_tension_for_frequency",
    "open_pipe_harmonic_frequency",
    "string_harmonic_frequency",
    "string_harmonic_wavelength",
]
