"""Linear momentum, impulse, and 1D collision formulas with Pint unit validation."""

from __future__ import annotations

import pint

from ..units import Quantity, QuantityLike, to_si, to_unit
from ..validation import require_range, validate_quantity


def linear_momentum(
    mass: QuantityLike,
    velocity: QuantityLike,
) -> pint.Quantity:
    """Compute linear momentum: ``p = m * v``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        velocity: Object velocity (dimension ``[velocity]``).

    Returns:
        Linear momentum as an SI base ``pint.Quantity`` (``kilogram * meter / second``).
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    v = validate_quantity(velocity, "[velocity]", name="velocity")
    return to_si(m * v)


def impulse(
    force: QuantityLike,
    duration: QuantityLike,
) -> pint.Quantity:
    """Compute impulse delivered by a constant or average force: ``J = F * delta_t``.

    Args:
        force: Applied force (dimension ``[force]``).
        duration: Time interval (dimension ``[time]``, ``>= 0``).

    Returns:
        Impulse as an SI base ``pint.Quantity`` (``kilogram * meter / second``).
    """
    f = validate_quantity(force, "[force]", name="force")
    dt = validate_quantity(duration, "[time]", name="duration", allow_negative=False)
    return to_si(f * dt)


def momentum_change(
    mass: QuantityLike,
    initial_velocity: QuantityLike,
    final_velocity: QuantityLike,
) -> pint.Quantity:
    """Compute change in linear momentum (equal to impulse): ``delta_p = m * (v_f - v_i)``.

    Args:
        mass: Object mass (dimension ``[mass]``, ``>= 0``).
        initial_velocity: Initial velocity (dimension ``[velocity]``).
        final_velocity: Final velocity (dimension ``[velocity]``).

    Returns:
        Momentum change as an SI base ``pint.Quantity`` (``kilogram * meter / second``).
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    v_i = validate_quantity(initial_velocity, "[velocity]", name="initial_velocity")
    v_f = validate_quantity(final_velocity, "[velocity]", name="final_velocity")
    return to_si(m * (v_f - v_i))


def average_force(
    impulse_value: QuantityLike,
    duration: QuantityLike,
) -> pint.Quantity:
    """Compute average net force from impulse and duration: ``F_avg = J / delta_t``.

    Args:
        impulse_value: Net impulse (compatible with ``N * s`` or ``kg * m / s``).
        duration: Collision or interaction duration (dimension ``[time]``, ``> 0``).

    Returns:
        Average force as a ``pint.Quantity`` in ``newton``.
    """
    j = validate_quantity(impulse_value, "N * s", name="impulse_value")
    dt = validate_quantity(
        duration,
        "[time]",
        name="duration",
        allow_negative=False,
        allow_zero=False,
    )
    return to_unit(j / dt, "N")


def collision_velocities(
    mass_1: QuantityLike,
    velocity_1: QuantityLike,
    mass_2: QuantityLike,
    velocity_2: QuantityLike,
    *,
    restitution: QuantityLike = 1.0,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Compute final velocities after a 1D collision with coefficient of restitution ``e``.

    Uses conservation of linear momentum together with ``v2' - v1' = -e * (v2 - v1)``.
    ``restitution=1.0`` corresponds to a perfectly elastic collision, while
    ``restitution=0.0`` corresponds to a perfectly inelastic collision.

    Args:
        mass_1: Mass of body 1 (dimension ``[mass]``, ``> 0``).
        velocity_1: Initial velocity of body 1 (dimension ``[velocity]``).
        mass_2: Mass of body 2 (dimension ``[mass]``, ``> 0``).
        velocity_2: Initial velocity of body 2 (dimension ``[velocity]``).
        restitution: Coefficient of restitution in ``[0, 1]`` (defaults to ``1.0``).

    Returns:
        Tuple ``(v1_final, v2_final)`` as SI ``pint.Quantity`` values in
        ``meter / second``.
    """
    m1 = validate_quantity(mass_1, "[mass]", name="mass_1", allow_negative=False, allow_zero=False)
    v1 = validate_quantity(velocity_1, "[velocity]", name="velocity_1")
    m2 = validate_quantity(mass_2, "[mass]", name="mass_2", allow_negative=False, allow_zero=False)
    v2 = validate_quantity(velocity_2, "[velocity]", name="velocity_2")
    e_q = validate_quantity(restitution, "dimensionless", name="restitution")
    require_range(
        e_q,
        min_value=Quantity(0.0, "dimensionless"),
        max_value=Quantity(1.0, "dimensionless"),
        name="restitution",
    )

    total_mass = m1 + m2
    total_momentum = m1 * v1 + m2 * v2
    v1_final = to_si((total_momentum + m2 * e_q * (v2 - v1)) / total_mass)
    v2_final = to_si((total_momentum + m1 * e_q * (v1 - v2)) / total_mass)
    return v1_final, v2_final


def elastic_collision_velocities(
    mass_1: QuantityLike,
    velocity_1: QuantityLike,
    mass_2: QuantityLike,
    velocity_2: QuantityLike,
) -> tuple[pint.Quantity, pint.Quantity]:
    """Compute final velocities after a 1D perfectly elastic collision (``e = 1``).

    Args:
        mass_1: Mass of body 1 (dimension ``[mass]``, ``> 0``).
        velocity_1: Initial velocity of body 1 (dimension ``[velocity]``).
        mass_2: Mass of body 2 (dimension ``[mass]``, ``> 0``).
        velocity_2: Initial velocity of body 2 (dimension ``[velocity]``).

    Returns:
        Tuple ``(v1_final, v2_final)`` as SI ``pint.Quantity`` values in
        ``meter / second``.
    """
    return collision_velocities(mass_1, velocity_1, mass_2, velocity_2, restitution=1.0)


def inelastic_collision_velocity(
    mass_1: QuantityLike,
    velocity_1: QuantityLike,
    mass_2: QuantityLike,
    velocity_2: QuantityLike,
) -> pint.Quantity:
    """Compute shared final velocity after a 1D perfectly inelastic collision (``e = 0``).

    Args:
        mass_1: Mass of body 1 (dimension ``[mass]``, ``> 0``).
        velocity_1: Initial velocity of body 1 (dimension ``[velocity]``).
        mass_2: Mass of body 2 (dimension ``[mass]``, ``> 0``).
        velocity_2: Initial velocity of body 2 (dimension ``[velocity]``).

    Returns:
        Combined final velocity as an SI ``pint.Quantity`` in ``meter / second``.
    """
    v_final, _ = collision_velocities(mass_1, velocity_1, mass_2, velocity_2, restitution=0.0)
    return v_final


__all__ = [
    "average_force",
    "collision_velocities",
    "elastic_collision_velocities",
    "impulse",
    "inelastic_collision_velocity",
    "linear_momentum",
    "momentum_change",
]
