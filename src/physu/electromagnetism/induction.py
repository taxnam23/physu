"""Electromagnetic induction formulas: flux, Faraday's law, inductors, RL/LC, and transformers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pint

from ..constants import VACUUM_PERMEABILITY
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import (
    _combine_additive,
    _combine_reciprocal,
    _cos_angle,
    _sin_angle,
    _to_clean_unit,
    _validate_positive_factor,
    _validate_turns,
)


def _validate_transformer_turns(
    primary_turns: QuantityLike,
    secondary_turns: QuantityLike,
) -> tuple[float, float]:
    """Validate positive primary and secondary winding turn counts."""
    n_p = _validate_turns(primary_turns, name="primary_turns")
    n_s = _validate_turns(secondary_turns, name="secondary_turns")
    return n_p, n_s


def magnetic_flux(
    magnetic_field: QuantityLike,
    area: QuantityLike,
    *,
    angle: QuantityLike = "0 deg",
) -> pint.Quantity:
    """Compute magnetic flux through a flat surface: ``Phi_B = B * A * cos(theta)``.

    Args:
        magnetic_field: Uniform magnetic flux density (dimension ``[magnetic_field]``).
        area: Surface area (dimension ``[area]``, ``>= 0``).
        angle: Angle between magnetic field and surface normal (compatible with
            ``"rad"``, defaults to ``"0 deg"``).

    Returns:
        Magnetic flux as a ``pint.Quantity`` in ``weber``.
    """
    b = validate_quantity(magnetic_field, "[magnetic_field]", name="magnetic_field")
    surf_area = validate_quantity(area, "[area]", name="area", allow_negative=False)
    cos_theta = _cos_angle(angle)
    return _to_clean_unit(b * surf_area * cos_theta, "Wb")


def faraday_emf(
    flux_change: QuantityLike,
    time_interval: QuantityLike,
    *,
    turns: QuantityLike = 1,
) -> pint.Quantity:
    """Compute average induced EMF via Faraday's law: ``emf = -N * delta_Phi_B / delta_t``.

    Args:
        flux_change: Change in magnetic flux per turn (dimension ``[magnetic_flux]``).
        time_interval: Positive elapsed time interval (dimension ``[time]``, ``> 0``).
        turns: Number of coil turns (dimensionless, ``> 0``, defaults to ``1``).

    Returns:
        Induced electromotive force (with Lenz sign) as a ``pint.Quantity`` in ``volt``.
    """
    dphi = validate_quantity(flux_change, "[magnetic_flux]", name="flux_change")
    dt = validate_quantity(
        time_interval,
        "[time]",
        name="time_interval",
        allow_negative=False,
        allow_zero=False,
    )
    n_turns = _validate_turns(turns)
    return _to_clean_unit(-n_turns * dphi / dt, "V")


def motional_emf(
    magnetic_field: QuantityLike,
    length: QuantityLike,
    speed: QuantityLike,
    *,
    angle: QuantityLike = "90 deg",
) -> pint.Quantity:
    """Compute motional EMF magnitude in a moving conductor: ``emf = B * L * v * |sin(theta)|``.

    Args:
        magnetic_field: Magnetic flux density (dimension ``[magnetic_field]``, ``>= 0``).
        length: Conductor length (dimension ``[length]``, ``>= 0``).
        speed: Conductor speed relative to field (dimension ``[velocity]``, ``>= 0``).
        angle: Angle between velocity and magnetic field (compatible with
            ``"rad"``, defaults to ``"90 deg"``).

    Returns:
        Motional EMF magnitude as a ``pint.Quantity`` in ``volt``.
    """
    b = validate_quantity(
        magnetic_field,
        "[magnetic_field]",
        name="magnetic_field",
        allow_negative=False,
    )
    length_q = validate_quantity(length, "[length]", name="length", allow_negative=False)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    sin_theta = abs(_sin_angle(angle))
    return _to_clean_unit(b * length_q * v * sin_theta, "V")


def solenoid_inductance(
    turns: QuantityLike,
    area: QuantityLike,
    length: QuantityLike,
    *,
    relative_permeability: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute self-inductance of an ideal solenoid: ``L = mu_0 * mu_r * N^2 * A / l``.

    Args:
        turns: Total coil turns (dimensionless, ``> 0``).
        area: Cross-sectional area of the solenoid core (dimension ``[area]``, ``>= 0``).
        length: Solenoid length (dimension ``[length]``, ``> 0``).
        relative_permeability: Core relative permeability (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Self-inductance as a ``pint.Quantity`` in ``henry``.
    """
    n_turns = _validate_turns(turns)
    area_q = validate_quantity(area, "[area]", name="area", allow_negative=False)
    length_q = validate_quantity(
        length,
        "[length]",
        name="length",
        allow_negative=False,
        allow_zero=False,
    )
    mu_r = _validate_positive_factor(
        relative_permeability,
        name="relative_permeability",
    )
    return _to_clean_unit(
        VACUUM_PERMEABILITY * mu_r * (n_turns**2) * area_q / length_q,
        "H",
    )


def inductor_energy(
    inductance: QuantityLike,
    current: QuantityLike,
) -> pint.Quantity:
    """Compute magnetic energy stored in an inductor: ``U_L = 0.5 * L * I^2``.

    Args:
        inductance: Self-inductance (dimension ``[inductance]``, ``>= 0``).
        current: Electric current through the inductor (dimension ``[current]``).

    Returns:
        Stored magnetic energy as a ``pint.Quantity`` in ``joule``.
    """
    l_q = validate_quantity(
        inductance,
        "[inductance]",
        name="inductance",
        allow_negative=False,
    )
    i = validate_quantity(current, "[current]", name="current")
    return _to_clean_unit(0.5 * l_q * (i**2), "J")


def series_inductance(inductances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent inductance for uncoupled inductors in series: ``L_eq = sum(L_i)``.

    Args:
        inductances: Non-empty sequence of non-negative inductances
            (dimension ``[inductance]``, each ``>= 0``).

    Returns:
        Equivalent inductance as a ``pint.Quantity`` in ``henry``.
    """
    return _combine_additive(
        inductances,
        "[inductance]",
        "H",
        item_label="inductances",
        allow_zero=True,
    )


def parallel_inductance(inductances: Sequence[QuantityLike]) -> pint.Quantity:
    """Compute equivalent parallel inductance (uncoupled): ``1 / L_eq = sum(1 / L_i)``.

    Args:
        inductances: Non-empty sequence of positive inductances
            (dimension ``[inductance]``, each ``> 0``).

    Returns:
        Equivalent inductance as a ``pint.Quantity`` in ``henry``.
    """
    return _combine_reciprocal(
        inductances,
        "[inductance]",
        "H",
        item_label="inductances",
    )


def rl_time_constant(
    inductance: QuantityLike,
    resistance: QuantityLike,
) -> pint.Quantity:
    """Compute RL circuit time constant: ``tau = L / R``.

    Args:
        inductance: Circuit inductance (dimension ``[inductance]``, ``>= 0``).
        resistance: Positive circuit resistance (dimension ``[resistance]``, ``> 0``).

    Returns:
        Time constant as a ``pint.Quantity`` in ``second``.
    """
    l_q = validate_quantity(
        inductance,
        "[inductance]",
        name="inductance",
        allow_negative=False,
    )
    r = validate_quantity(
        resistance,
        "[resistance]",
        name="resistance",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(l_q / r, "s")


def lc_resonant_frequency(
    inductance: QuantityLike,
    capacitance: QuantityLike,
) -> pint.Quantity:
    """Compute natural resonant frequency of an LC tank: ``f_0 = 1 / (2 * pi * sqrt(L * C))``.

    Args:
        inductance: Positive inductance (dimension ``[inductance]``, ``> 0``).
        capacitance: Positive capacitance (dimension ``[capacitance]``, ``> 0``).

    Returns:
        Resonant frequency as a ``pint.Quantity`` in ``hertz``.
    """
    l_q = validate_quantity(
        inductance,
        "[inductance]",
        name="inductance",
        allow_negative=False,
        allow_zero=False,
    )
    c_q = validate_quantity(
        capacitance,
        "[capacitance]",
        name="capacitance",
        allow_negative=False,
        allow_zero=False,
    )
    lc_s2 = float(to_si(l_q * c_q).magnitude)
    return Quantity(1.0 / (2.0 * math.pi * math.sqrt(lc_s2)), "Hz")


def transformer_voltage(
    primary_voltage: QuantityLike,
    primary_turns: QuantityLike,
    secondary_turns: QuantityLike,
) -> pint.Quantity:
    """Compute ideal transformer secondary voltage: ``V_s = V_p * (N_s / N_p)``.

    Args:
        primary_voltage: Primary winding voltage (dimension ``[electric_potential]``).
        primary_turns: Primary winding turns (dimensionless, ``> 0``).
        secondary_turns: Secondary winding turns (dimensionless, ``> 0``).

    Returns:
        Secondary voltage as a ``pint.Quantity`` in ``volt``.
    """
    v_p = validate_quantity(
        primary_voltage,
        "[electric_potential]",
        name="primary_voltage",
    )
    n_p, n_s = _validate_transformer_turns(primary_turns, secondary_turns)
    return _to_clean_unit(v_p * (n_s / n_p), "V")


def transformer_current(
    primary_current: QuantityLike,
    primary_turns: QuantityLike,
    secondary_turns: QuantityLike,
) -> pint.Quantity:
    """Compute ideal transformer secondary current: ``I_s = I_p * (N_p / N_s)``.

    Args:
        primary_current: Primary winding current (dimension ``[current]``).
        primary_turns: Primary winding turns (dimensionless, ``> 0``).
        secondary_turns: Secondary winding turns (dimensionless, ``> 0``).

    Returns:
        Secondary current as a ``pint.Quantity`` in ``ampere``.
    """
    i_p = validate_quantity(primary_current, "[current]", name="primary_current")
    n_p, n_s = _validate_transformer_turns(primary_turns, secondary_turns)
    return _to_clean_unit(i_p * (n_p / n_s), "A")


__all__ = [
    "faraday_emf",
    "inductor_energy",
    "lc_resonant_frequency",
    "magnetic_flux",
    "motional_emf",
    "parallel_inductance",
    "rl_time_constant",
    "series_inductance",
    "solenoid_inductance",
    "transformer_current",
    "transformer_voltage",
]
