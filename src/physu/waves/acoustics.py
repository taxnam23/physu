"""Acoustics formulas: ideal-gas sound speed, decibel levels, and acoustic impedance."""

from __future__ import annotations

import math

import pint

from ..exceptions import PhysicalDomainError
from ..thermodynamics.ideal_gas import rms_molecular_speed
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _to_clean_unit, _validate_logarithmic_level


def _decibel_level(
    value: QuantityLike,
    reference: QuantityLike,
    *,
    dimension: str,
    factor: float,
    value_name: str,
    ref_name: str,
) -> pint.Quantity:
    """Validate positive value/reference and compute ``factor * log10(value / reference)`` in dB."""
    val_q = validate_quantity(
        value,
        dimension,
        name=value_name,
        allow_negative=False,
        allow_zero=False,
    )
    ref_q = validate_quantity(
        reference,
        dimension,
        name=ref_name,
        allow_negative=False,
        allow_zero=False,
    )
    ratio = float(to_si(val_q / ref_q).magnitude)
    level_db = factor * math.log10(ratio)
    return Quantity(0.0 if level_db == 0 else level_db, "dB")


def speed_of_sound_ideal_gas(
    adiabatic_index: QuantityLike,
    temperature: QuantityLike,
    molar_mass: QuantityLike,
) -> pint.Quantity:
    """Compute speed of sound in an ideal gas: ``v = sqrt(gamma * R * T / M)``.

    Args:
        adiabatic_index: Ratio of specific heats ``gamma = C_p / C_v``
            (dimensionless, ``> 0``).
        temperature: Absolute gas temperature (``>= 0 K``).
        molar_mass: Gas molar mass (compatible with ``kg / mol``, ``> 0``).

    Returns:
        Speed of sound as a ``pint.Quantity`` in ``meter / second``.
    """
    gamma = float(
        validate_quantity(
            adiabatic_index,
            "dimensionless",
            name="adiabatic_index",
            allow_negative=False,
            allow_zero=False,
        )
        .to("dimensionless")
        .magnitude
    )
    v_rms = rms_molecular_speed(temperature, molar_mass)
    speed = math.sqrt(gamma / 3.0) * float(v_rms.magnitude)
    return Quantity(0.0 if speed == 0 else speed, "m / s")


def sound_intensity_level(
    intensity: QuantityLike,
    *,
    reference_intensity: QuantityLike = "1e-12 W / m^2",
) -> pint.Quantity:
    """Compute sound intensity level in decibels: ``beta = 10 * log10(I / I_0)``.

    Args:
        intensity: Sound wave intensity (compatible with ``W / m^2``, ``> 0``).
        reference_intensity: Reference threshold intensity (compatible with
            ``W / m^2``, ``> 0``, defaults to ``"1e-12 W / m^2"``).

    Returns:
        Sound intensity level as a ``pint.Quantity`` in ``decibel`` (``dB``).
    """
    return _decibel_level(
        intensity,
        reference_intensity,
        dimension="W / m^2",
        factor=10.0,
        value_name="intensity",
        ref_name="reference_intensity",
    )


def intensity_from_sound_level(
    sound_level: QuantityLike,
    *,
    reference_intensity: QuantityLike = "1e-12 W / m^2",
) -> pint.Quantity:
    """Compute sound intensity from a decibel or neper level: ``I = I_0 * 10^(beta / 10)``.

    Args:
        sound_level: Sound intensity level in ``dB`` (or ``Np``).
        reference_intensity: Reference threshold intensity (compatible with
            ``W / m^2``, ``> 0``, defaults to ``"1e-12 W / m^2"``).

    Returns:
        Sound wave intensity as a ``pint.Quantity`` in ``watt / meter ** 2``.
    """
    level_q = _validate_logarithmic_level(sound_level, name="sound_level")
    i0 = validate_quantity(
        reference_intensity,
        "W / m^2",
        name="reference_intensity",
        allow_negative=False,
        allow_zero=False,
    )
    beta_db = float(level_q.to("dB").magnitude)
    ratio = 10.0 ** (beta_db / 10.0)
    return _to_clean_unit(i0 * ratio, "W / m^2")


def sound_pressure_level(
    pressure_rms: QuantityLike,
    *,
    reference_pressure: QuantityLike = "20 uPa",
) -> pint.Quantity:
    """Compute sound pressure level in decibels: ``L_p = 20 * log10(p_rms / p_0)``.

    Args:
        pressure_rms: Root-mean-square acoustic pressure (``[pressure]``, ``> 0``).
        reference_pressure: Reference acoustic pressure (``[pressure]``, ``> 0``,
            defaults to ``"20 uPa"``).

    Returns:
        Sound pressure level as a ``pint.Quantity`` in ``decibel`` (``dB``).
    """
    return _decibel_level(
        pressure_rms,
        reference_pressure,
        dimension="[pressure]",
        factor=20.0,
        value_name="pressure_rms",
        ref_name="reference_pressure",
    )


def acoustic_impedance(
    medium_density: QuantityLike,
    sound_speed: QuantityLike,
) -> pint.Quantity:
    """Compute specific acoustic impedance: ``Z = rho * c``.

    Args:
        medium_density: Medium mass density (compatible with ``kg / m^3``, ``>= 0``).
        sound_speed: Speed of sound in the medium (dimension ``[velocity]``, ``>= 0``).

    Returns:
        Specific acoustic impedance as a ``pint.Quantity`` in
        ``pascal * second / meter`` (rayl).
    """
    rho = validate_quantity(
        medium_density,
        "kg / m^3",
        name="medium_density",
        allow_negative=False,
    )
    c = validate_quantity(sound_speed, "[velocity]", name="sound_speed", allow_negative=False)
    return _to_clean_unit(rho * c, "Pa * s / m")


def acoustic_reflection_coefficient(
    impedance_1: QuantityLike,
    impedance_2: QuantityLike,
) -> pint.Quantity:
    """Compute normal-incidence intensity reflection coefficient: ``R = ((Z2 - Z1) / (Z2 + Z1))^2``.

    Args:
        impedance_1: Acoustic impedance of incident medium (compatible with
            ``Pa * s / m``, ``>= 0``).
        impedance_2: Acoustic impedance of transmitted medium (compatible with
            ``Pa * s / m``, ``>= 0``).

    Returns:
        Dimensionless intensity reflection coefficient in ``[0, 1]``.

    Raises:
        PhysicalDomainError: If ``impedance_1 + impedance_2 == 0``.
    """
    z1 = validate_quantity(impedance_1, "Pa * s / m", name="impedance_1", allow_negative=False)
    z2 = validate_quantity(impedance_2, "Pa * s / m", name="impedance_2", allow_negative=False)
    total_z = z1 + z2
    if float(total_z.magnitude) <= 0:
        raise PhysicalDomainError(
            "Sum of acoustic impedances (impedance_1 + impedance_2) must be positive."
        )
    return to_si(((z2 - z1) / total_z) ** 2)


__all__ = [
    "acoustic_impedance",
    "acoustic_reflection_coefficient",
    "intensity_from_sound_level",
    "sound_intensity_level",
    "sound_pressure_level",
    "speed_of_sound_ideal_gas",
]
