"""Mechanical energy conservation, escape speed, and efficiency formulas."""

from __future__ import annotations

import math

import pint

from ..constants import GRAVITATIONAL_CONSTANT, STANDARD_GRAVITY
from ..dynamics.gravitation import orbital_speed
from ..exceptions import PhysicalDomainError
from ..kinematics.accelerated import torricelli_velocity
from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import validate_quantity


def _validate_initial_state(
    initial_speed: QuantityLike,
    initial_height: QuantityLike,
    gravity: QuantityLike,
) -> tuple[pint.Quantity, pint.Quantity, pint.Quantity]:
    """Validate non-negative initial speed, initial height, and positive gravity."""
    v_i = validate_quantity(
        initial_speed,
        "[velocity]",
        name="initial_speed",
        allow_negative=False,
    )
    h_i = validate_quantity(initial_height, "[length]", name="initial_height")
    g = validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )
    return v_i, h_i, g


def mechanical_energy(
    kinetic: QuantityLike,
    potential: QuantityLike,
) -> pint.Quantity:
    """Compute total mechanical energy: ``E = K + U``.

    Args:
        kinetic: Kinetic energy (dimension ``[energy]``, ``>= 0``).
        potential: Potential energy (dimension ``[energy]``).

    Returns:
        Total mechanical energy as a ``pint.Quantity`` in ``joule``.
    """
    k = validate_quantity(kinetic, "[energy]", name="kinetic", allow_negative=False)
    u = validate_quantity(potential, "[energy]", name="potential")
    return to_unit(k + u, "J")


def non_conservative_work(
    initial_energy: QuantityLike,
    final_energy: QuantityLike,
) -> pint.Quantity:
    """Compute work done by non-conservative forces: ``W_nc = E_f - E_i``.

    Args:
        initial_energy: Initial total mechanical energy (dimension ``[energy]``).
        final_energy: Final total mechanical energy (dimension ``[energy]``).

    Returns:
        Non-conservative work as a ``pint.Quantity`` in ``joule``.
    """
    e_i = validate_quantity(initial_energy, "[energy]", name="initial_energy")
    e_f = validate_quantity(final_energy, "[energy]", name="final_energy")
    return to_unit(e_f - e_i, "J")


def final_speed_conservation(
    initial_speed: QuantityLike,
    initial_height: QuantityLike,
    final_height: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute final speed under mechanical energy conservation in uniform gravity.

    Solves ``0.5 * v_i^2 + g * h_i = 0.5 * v_f^2 + g * h_f``.

    Args:
        initial_speed: Initial speed magnitude (dimension ``[velocity]``, ``>= 0``).
        initial_height: Initial height (dimension ``[length]``).
        final_height: Target height (dimension ``[length]``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Final speed magnitude as an SI base ``pint.Quantity`` (``meter / second``).

    Raises:
        PhysicalDomainError: If ``final_height`` exceeds maximum reachable height.
    """
    v_i, h_i, g = _validate_initial_state(initial_speed, initial_height, gravity)
    h_f = validate_quantity(final_height, "[length]", name="final_height")
    return torricelli_velocity(v_i, -g, h_f - h_i)


def final_height_conservation(
    initial_speed: QuantityLike,
    initial_height: QuantityLike,
    final_speed: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute final height under mechanical energy conservation in uniform gravity.

    Solves ``h_f = h_i + (v_i^2 - v_f^2) / (2 * g)``.

    Args:
        initial_speed: Initial speed magnitude (dimension ``[velocity]``, ``>= 0``).
        initial_height: Initial height (dimension ``[length]``).
        final_speed: Final speed magnitude (dimension ``[velocity]``, ``>= 0``).
        gravity: Gravitational acceleration magnitude (``> 0``, defaults to
            ``STANDARD_GRAVITY``).

    Returns:
        Final height as an SI base ``pint.Quantity`` (``meter``).
    """
    v_i, h_i, g = _validate_initial_state(initial_speed, initial_height, gravity)
    v_f = validate_quantity(
        final_speed,
        "[velocity]",
        name="final_speed",
        allow_negative=False,
    )
    return to_si(h_i + ((v_i**2) - (v_f**2)) / (2.0 * g))


def escape_speed(
    central_mass: QuantityLike,
    radius: QuantityLike,
    *,
    gravitational_constant: QuantityLike = GRAVITATIONAL_CONSTANT,
) -> pint.Quantity:
    """Compute gravitational escape speed: ``v_e = sqrt(2 * G * M / r)``.

    Args:
        central_mass: Central body mass (dimension ``[mass]``, ``>= 0``).
        radius: Distance from body center (dimension ``[length]``, ``> 0``).
        gravitational_constant: Universal gravitational constant (compatible with
            ``m^3 / (kg * s^2)``, ``> 0``, defaults to ``GRAVITATIONAL_CONSTANT``).

    Returns:
        Escape speed as an SI base ``pint.Quantity`` (``meter / second``).
    """
    v_orb = orbital_speed(
        central_mass,
        radius,
        gravitational_constant=gravitational_constant,
    )
    return Quantity(math.sqrt(2.0) * float(v_orb.to("m / s").magnitude), "m / s")


def mechanical_efficiency(
    useful_energy: QuantityLike,
    input_energy: QuantityLike,
) -> pint.Quantity:
    """Compute mechanical efficiency: ``eta = E_out / E_in``.

    Args:
        useful_energy: Useful energy output (dimension ``[energy]``, ``>= 0``).
        input_energy: Total energy input (dimension ``[energy]``, ``> 0``).

    Returns:
        Dimensionless efficiency in ``[0, 1]`` as a ``pint.Quantity`` (convertible
        to ``"%"``).

    Raises:
        PhysicalDomainError: If ``useful_energy > input_energy`` or inputs violate
            sign constraints.
    """
    e_out = validate_quantity(
        useful_energy,
        "[energy]",
        name="useful_energy",
        allow_negative=False,
    )
    e_in = validate_quantity(
        input_energy,
        "[energy]",
        name="input_energy",
        allow_negative=False,
        allow_zero=False,
    )
    eta = to_si(e_out / e_in)
    if float(eta.magnitude) > 1.0:
        raise PhysicalDomainError(
            "Useful output energy cannot exceed total input energy "
            f"(got efficiency {eta})."
        )
    return eta


__all__ = [
    "escape_speed",
    "final_height_conservation",
    "final_speed_conservation",
    "mechanical_efficiency",
    "mechanical_energy",
    "non_conservative_work",
]
