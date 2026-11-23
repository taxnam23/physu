"""Nuclear physics and radioactivity formulas: binding energy, half-life, decay, and dating."""

from __future__ import annotations

import math

import pint

from ..constants import SPEED_OF_LIGHT
from ..exceptions import PhysicalDomainError
from ..units import QuantityLike, check_dimension, to_si
from ..validation import require_non_negative, require_positive, validate_quantity
from ._helpers import _resolve_decay_constant, _to_clean_unit
from .relativity import rest_energy


def mass_energy_equivalence(
    mass_defect: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute nuclear binding or reaction energy from mass defect: ``E = delta_m * c^2``.

    Args:
        mass_defect: Mass defect ``delta_m`` (dimension ``[mass]``, ``>= 0``;
            e.g. in ``"u"`` or ``"kg"``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Equivalent energy as a ``pint.Quantity`` in ``joule`` (convertible to ``MeV``).
    """
    dm = validate_quantity(mass_defect, "[mass]", name="mass_defect", allow_negative=False)
    return rest_energy(dm, speed_of_light=speed_of_light)


def decay_constant(half_life: QuantityLike) -> pint.Quantity:
    """Compute exponential radioactive decay constant: ``lambda = ln(2) / t_half``.

    Args:
        half_life: Positive isotope half-life (dimension ``[time]``, ``> 0``).

    Returns:
        Decay constant as a ``pint.Quantity`` in ``1 / second``.
    """
    return _resolve_decay_constant(half_life, None)


def half_life_from_decay_constant(decay_constant: QuantityLike) -> pint.Quantity:
    """Compute isotope half-life from decay constant: ``t_half = ln(2) / lambda``.

    Args:
        decay_constant: Positive decay constant (compatible with ``1 / s``, ``> 0``).

    Returns:
        Half-life as a ``pint.Quantity`` in ``second`` (convertible to ``year``).
    """
    lam = validate_quantity(
        decay_constant,
        "1 / s",
        name="decay_constant",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit(math.log(2.0) / lam, "s")


def radioactive_remaining(
    initial_amount: QuantityLike,
    elapsed_time: QuantityLike,
    *,
    half_life: QuantityLike | None = None,
    decay_constant: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute remaining radioactive nuclei, mass, or activity: ``N(t) = N_0 * exp(-lambda * t)``.

    Provide exactly one of ``half_life`` or ``decay_constant``.

    Args:
        initial_amount: Initial non-negative quantity (count, mass, moles, or activity).
        elapsed_time: Non-negative elapsed time (dimension ``[time]``, ``>= 0``).
        half_life: Positive isotope half-life (dimension ``[time]``, ``> 0``).
        decay_constant: Positive decay constant (compatible with ``1 / s``, ``> 0``).

    Returns:
        Remaining amount as a ``pint.Quantity`` in the units of ``initial_amount``.
    """
    n0 = require_non_negative(initial_amount, name="initial_amount")
    t = validate_quantity(elapsed_time, "[time]", name="elapsed_time", allow_negative=False)
    lam = _resolve_decay_constant(half_life, decay_constant)
    exponent = float(to_si(lam * t).magnitude)
    return _to_clean_unit(n0 * math.exp(-exponent), str(n0.units))


def radioactive_activity(
    sample_count: QuantityLike,
    *,
    half_life: QuantityLike | None = None,
    decay_constant: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute instantaneous radioactive activity: ``A = lambda * N``.

    Provide exactly one of ``half_life`` or ``decay_constant``.

    Args:
        sample_count: Number of undecayed nuclei ``N`` (dimensionless, ``>= 0``).
        half_life: Positive isotope half-life (dimension ``[time]``, ``> 0``).
        decay_constant: Positive decay constant (compatible with ``1 / s``, ``> 0``).

    Returns:
        Radioactive activity as a ``pint.Quantity`` in ``becquerel``
        (convertible to ``curie``).
    """
    n_count = validate_quantity(
        sample_count,
        "dimensionless",
        name="sample_count",
        allow_negative=False,
    )
    lam = _resolve_decay_constant(half_life, decay_constant)
    return _to_clean_unit(lam * n_count, "Bq")


def radiometric_age(
    initial_amount: QuantityLike,
    remaining_amount: QuantityLike,
    *,
    half_life: QuantityLike | None = None,
    decay_constant: QuantityLike | None = None,
) -> pint.Quantity:
    """Compute elapsed age via radiometric dating: ``t = ln(N_0 / N) / lambda``.

    Provide exactly one of ``half_life`` or ``decay_constant``.

    Args:
        initial_amount: Positive initial amount ``N_0`` (count, mass, moles, or activity).
        remaining_amount: Positive remaining amount ``N <= N_0`` with compatible units.
        half_life: Positive isotope half-life (dimension ``[time]``, ``> 0``).
        decay_constant: Positive decay constant (compatible with ``1 / s``, ``> 0``).

    Returns:
        Elapsed sample age as a ``pint.Quantity`` in ``second`` (convertible to ``year``).

    Raises:
        PhysicalDomainError: If ``remaining_amount > initial_amount``.
    """
    n0 = require_positive(initial_amount, name="initial_amount")
    n_rem = require_positive(remaining_amount, name="remaining_amount")
    check_dimension(n_rem, n0.units, name="remaining_amount")
    ratio = float(to_si(n0 / n_rem).magnitude)
    if ratio < 1.0:
        raise PhysicalDomainError(
            "Parameter 'remaining_amount' cannot exceed 'initial_amount'."
        )
    lam = _resolve_decay_constant(half_life, decay_constant)
    return _to_clean_unit(math.log(ratio) / lam, "s")


__all__ = [
    "decay_constant",
    "half_life_from_decay_constant",
    "mass_energy_equivalence",
    "radioactive_activity",
    "radioactive_remaining",
    "radiometric_age",
]
