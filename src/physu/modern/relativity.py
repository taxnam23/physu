"""Special relativity formulas: Lorentz factor, dilation, contraction, energy, and momentum."""

from __future__ import annotations

import math

import pint

from ..constants import SPEED_OF_LIGHT
from ..units import Quantity, QuantityLike
from ..validation import validate_quantity
from ._helpers import _subluminal_beta, _to_clean_unit, _validate_positive_c


def lorentz_factor(
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute relativistic Lorentz factor: ``gamma = 1 / sqrt(1 - (v / c)^2)``.

    Args:
        speed: Relative speed or signed velocity (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Dimensionless Lorentz factor ``>= 1`` as a ``pint.Quantity``.
    """
    beta = _subluminal_beta(speed, speed_of_light, speed_name="speed")
    return Quantity(1.0 / math.sqrt(1.0 - (beta**2)), "dimensionless")


def time_dilation(
    proper_time: QuantityLike,
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute dilated coordinate time interval: ``delta_t = gamma * delta_t_0``.

    Args:
        proper_time: Proper time interval in the rest frame (``[time]``, ``>= 0``).
        speed: Relative frame speed (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Dilated time interval as a ``pint.Quantity`` in ``second``.
    """
    dt0 = validate_quantity(proper_time, "[time]", name="proper_time", allow_negative=False)
    gamma = lorentz_factor(speed, speed_of_light=speed_of_light)
    return _to_clean_unit(dt0 * gamma, "s")


def length_contraction(
    proper_length: QuantityLike,
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute Lorentz-contracted length along motion: ``L = L_0 / gamma``.

    Args:
        proper_length: Rest-frame proper length (dimension ``[length]``, ``>= 0``).
        speed: Relative frame speed (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Contracted length as a ``pint.Quantity`` in ``meter``.
    """
    l0 = validate_quantity(proper_length, "[length]", name="proper_length", allow_negative=False)
    gamma = lorentz_factor(speed, speed_of_light=speed_of_light)
    return _to_clean_unit(l0 / gamma, "m")


def relativistic_velocity_addition(
    velocity_1: QuantityLike,
    velocity_2: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute relativistic velocity addition: ``u = (v_1 + v_2) / (1 + v_1 * v_2 / c^2)``.

    Args:
        velocity_1: First velocity along the line of motion (``[velocity]``, ``|v_1| < c``).
        velocity_2: Second velocity along the line of motion (``[velocity]``, ``|v_2| < c``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Combined velocity as a ``pint.Quantity`` in ``meter / second``.
    """
    c = _validate_positive_c(speed_of_light)
    beta1 = _subluminal_beta(velocity_1, c, speed_name="velocity_1")
    beta2 = _subluminal_beta(velocity_2, c, speed_name="velocity_2")
    combined_beta = (beta1 + beta2) / (1.0 + (beta1 * beta2))
    return _to_clean_unit(combined_beta * c, "m / s")


def rest_energy(
    mass: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute mass-energy equivalence at rest: ``E_0 = m * c^2``.

    Args:
        mass: Particle or system invariant mass (dimension ``[mass]``, ``>= 0``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Rest energy as a ``pint.Quantity`` in ``joule`` (convertible to ``MeV``).
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    c = _validate_positive_c(speed_of_light)
    return _to_clean_unit(m * (c**2), "J")


def relativistic_total_energy(
    mass: QuantityLike,
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute total relativistic energy: ``E = gamma * m * c^2``.

    Args:
        mass: Particle invariant mass (dimension ``[mass]``, ``>= 0``).
        speed: Particle speed (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Total energy as a ``pint.Quantity`` in ``joule``.
    """
    e0 = rest_energy(mass, speed_of_light=speed_of_light)
    gamma = lorentz_factor(speed, speed_of_light=speed_of_light)
    return _to_clean_unit(e0 * gamma, "J")


def relativistic_kinetic_energy(
    mass: QuantityLike,
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute relativistic kinetic energy: ``K = (gamma - 1) * m * c^2``.

    Args:
        mass: Particle invariant mass (dimension ``[mass]``, ``>= 0``).
        speed: Particle speed (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Relativistic kinetic energy as a ``pint.Quantity`` in ``joule``.
    """
    e0 = rest_energy(mass, speed_of_light=speed_of_light)
    gamma = lorentz_factor(speed, speed_of_light=speed_of_light)
    return _to_clean_unit(e0 * (float(gamma.magnitude) - 1.0), "J")


def relativistic_momentum(
    mass: QuantityLike,
    speed: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute relativistic linear momentum: ``p = gamma * m * v``.

    Args:
        mass: Particle invariant mass (dimension ``[mass]``, ``>= 0``).
        speed: Signed particle velocity (``[velocity]``, ``|v| < speed_of_light``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Relativistic momentum as a ``pint.Quantity`` in ``kilogram * meter / second``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    c = _validate_positive_c(speed_of_light)
    beta = _subluminal_beta(speed, c, speed_name="speed")
    gamma = 1.0 / math.sqrt(1.0 - (beta**2))
    v = beta * c
    return _to_clean_unit(gamma * m * v, "kg * m / s")


def energy_momentum_relation(
    mass: QuantityLike,
    momentum: QuantityLike,
    *,
    speed_of_light: QuantityLike = SPEED_OF_LIGHT,
) -> pint.Quantity:
    """Compute total energy from mass and momentum: ``E = sqrt((p * c)^2 + (m * c^2)^2)``.

    Args:
        mass: Particle invariant mass (dimension ``[mass]``, ``>= 0``; ``0`` for photons).
        momentum: Linear momentum (compatible with ``kg * m / s``).
        speed_of_light: Speed of light in vacuum (``[velocity]``, ``> 0``,
            defaults to ``SPEED_OF_LIGHT``).

    Returns:
        Total energy as a ``pint.Quantity`` in ``joule``.
    """
    e0_j = float(rest_energy(mass, speed_of_light=speed_of_light).to("J").magnitude)
    p = validate_quantity(momentum, "kg * m / s", name="momentum")
    c = _validate_positive_c(speed_of_light)
    pc_j = float(_to_clean_unit(p * c, "J").magnitude)
    return Quantity(math.hypot(pc_j, e0_j), "J")


__all__ = [
    "energy_momentum_relation",
    "length_contraction",
    "lorentz_factor",
    "relativistic_kinetic_energy",
    "relativistic_momentum",
    "relativistic_total_energy",
    "relativistic_velocity_addition",
    "rest_energy",
    "time_dilation",
]
