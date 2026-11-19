"""Shared internal validation and trigonometric helpers for physu.waves."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pint

from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_unit, ureg
from ..validation import validate_quantity

_TRIG_ZERO_TOL = 1e-15


def _to_clean_unit(quantity: Any, target_unit: str) -> pint.Quantity:
    """Convert ``quantity`` to ``target_unit`` and normalize ``-0.0`` to ``0.0``."""
    result = to_unit(quantity, target_unit)
    return Quantity(0.0, target_unit) if result.magnitude == 0 else result


def _snap_zero(value: float) -> float:
    """Snap near-zero floating point artifacts to exact ``0.0``."""
    return 0.0 if abs(value) < _TRIG_ZERO_TOL else value


def _eval_phase_trig(
    angular_frequency: QuantityLike,
    time: QuantityLike,
    phase: QuantityLike,
    trig_fn: Callable[[float], float],
) -> tuple[pint.Quantity, float]:
    """Validate SHM phase inputs and evaluate ``trig_fn(omega * t + phase)``."""
    omega = validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
    t = validate_quantity(time, "[time]", name="time", allow_negative=False)
    phi = validate_quantity(phase, "rad", name="phase")
    total_phase = float((omega * t + phi).to("rad").magnitude)
    return omega, _snap_zero(trig_fn(total_phase))


def _validate_harmonic(harmonic: QuantityLike, *, odd_only: bool = False) -> int:
    """Validate a positive integer harmonic number (and oddness when required)."""
    q = validate_quantity(
        harmonic,
        "dimensionless",
        name="harmonic",
        allow_negative=False,
        allow_zero=False,
    )
    mag = float(q.to("dimensionless").magnitude)
    if not mag.is_integer():
        raise PhysicalDomainError(
            f"Parameter 'harmonic' must be a positive integer, got {harmonic}."
        )
    n = int(mag)
    if odd_only and n % 2 == 0:
        raise PhysicalDomainError(
            f"Closed-open pipe only supports odd harmonics, got {n}."
        )
    return n


def _validate_logarithmic_level(
    level: QuantityLike,
    *,
    name: str = "sound_level",
) -> pint.Quantity:
    """Validate a logarithmic level expressed in ``dB`` (``decibel``) or ``Np`` (``neper``)."""
    q = validate_quantity(level, "dimensionless", name=name)
    if q.units not in (ureg.decibel, ureg.neper):
        raise PhysicalDomainError(
            f"Parameter '{name}' must use a logarithmic unit ('dB' or 'Np'), "
            f"got '{q.units}'."
        )
    return q
