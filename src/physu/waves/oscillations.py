"""Simple harmonic motion (SHM), spring-mass oscillators, and pendulum formulas."""

from __future__ import annotations

import math

import pint

from ..constants import STANDARD_GRAVITY
from ..energy.potential import elastic_potential_energy
from ..exceptions import PhysicalDomainError
from ..units import Quantity, QuantityLike, to_si
from ..validation import validate_quantity
from ._helpers import _eval_phase_trig, _to_clean_unit


def _validate_positive_gravity(gravity: QuantityLike) -> pint.Quantity:
    """Validate strictly positive gravitational acceleration."""
    return validate_quantity(
        gravity,
        "[acceleration]",
        name="gravity",
        allow_negative=False,
        allow_zero=False,
    )


def _two_pi_sqrt_seconds(squared_time: pint.Quantity) -> pint.Quantity:
    """Compute ``2 * pi * sqrt(squared_time)`` in ``second``."""
    t_sq = float(to_si(squared_time).magnitude)
    return Quantity(2.0 * math.pi * math.sqrt(t_sq), "s")


def spring_mass_angular_frequency(
    spring_constant: QuantityLike,
    mass: QuantityLike,
) -> pint.Quantity:
    """Compute natural angular frequency of a spring-mass oscillator: ``omega = sqrt(k / m)``.

    Args:
        spring_constant: Spring stiffness (compatible with ``N / m``, ``>= 0``).
        mass: Oscillating mass (dimension ``[mass]``, ``> 0``).

    Returns:
        Angular frequency as a ``pint.Quantity`` in ``radian / second``.
    """
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
    )
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    omega_sq = float(to_si(k / m).magnitude)
    return Quantity(math.sqrt(omega_sq), "rad / s")


def spring_mass_period(
    mass: QuantityLike,
    spring_constant: QuantityLike,
) -> pint.Quantity:
    """Compute oscillation period of a spring-mass system: ``T = 2 * pi * sqrt(m / k)``.

    Args:
        mass: Oscillating mass (dimension ``[mass]``, ``>= 0``).
        spring_constant: Positive spring stiffness (compatible with ``N / m``, ``> 0``).

    Returns:
        Oscillation period as a ``pint.Quantity`` in ``second``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    k = validate_quantity(
        spring_constant,
        "N / m",
        name="spring_constant",
        allow_negative=False,
        allow_zero=False,
    )
    return _two_pi_sqrt_seconds(m / k)


def simple_pendulum_period(
    length: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute small-angle period of a simple pendulum: ``T = 2 * pi * sqrt(L / g)``.

    Args:
        length: Pendulum length (dimension ``[length]``, ``>= 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Pendulum period as a ``pint.Quantity`` in ``second``.
    """
    length_q = validate_quantity(length, "[length]", name="length", allow_negative=False)
    g = _validate_positive_gravity(gravity)
    return _two_pi_sqrt_seconds(length_q / g)


def physical_pendulum_period(
    moment_of_inertia: QuantityLike,
    mass: QuantityLike,
    pivot_distance: QuantityLike,
    *,
    gravity: QuantityLike = STANDARD_GRAVITY,
) -> pint.Quantity:
    """Compute small-angle period of a physical pendulum: ``T = 2 * pi * sqrt(I / (m * g * d))``.

    Args:
        moment_of_inertia: Moment of inertia about the pivot (compatible with
            ``kg * m^2``, ``>= 0``).
        mass: Total body mass (dimension ``[mass]``, ``> 0``).
        pivot_distance: Distance from pivot to center of mass (``[length]``, ``> 0``).
        gravity: Gravitational acceleration (dimension ``[acceleration]``, ``> 0``,
            defaults to ``STANDARD_GRAVITY``).

    Returns:
        Pendulum period as a ``pint.Quantity`` in ``second``.
    """
    inertia = validate_quantity(
        moment_of_inertia,
        "kg * m^2",
        name="moment_of_inertia",
        allow_negative=False,
    )
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    d = validate_quantity(
        pivot_distance,
        "[length]",
        name="pivot_distance",
        allow_negative=False,
        allow_zero=False,
    )
    g = _validate_positive_gravity(gravity)
    return _two_pi_sqrt_seconds(inertia / (m * g * d))


def shm_displacement(
    amplitude: QuantityLike,
    angular_frequency: QuantityLike,
    time: QuantityLike,
    *,
    phase: QuantityLike = "0 rad",
) -> pint.Quantity:
    """Compute SHM displacement: ``x(t) = A * cos(omega * t + phase)``.

    Args:
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).
        time: Elapsed time (dimension ``[time]``, ``>= 0``).
        phase: Initial phase constant (compatible with ``"rad"``, defaults to ``"0 rad"``).

    Returns:
        Instantaneous displacement as a ``pint.Quantity`` in ``meter``.
    """
    amp = validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
    _, cos_phase = _eval_phase_trig(angular_frequency, time, phase, math.cos)
    return _to_clean_unit(amp * cos_phase, "m")


def shm_velocity(
    amplitude: QuantityLike,
    angular_frequency: QuantityLike,
    time: QuantityLike,
    *,
    phase: QuantityLike = "0 rad",
) -> pint.Quantity:
    """Compute SHM velocity: ``v(t) = -A * omega * sin(omega * t + phase)``.

    Args:
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).
        time: Elapsed time (dimension ``[time]``, ``>= 0``).
        phase: Initial phase constant (compatible with ``"rad"``, defaults to ``"0 rad"``).

    Returns:
        Instantaneous velocity as a ``pint.Quantity`` in ``meter / second``.
    """
    amp_m = float(
        validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
        .to("m")
        .magnitude
    )
    omega, sin_phase = _eval_phase_trig(angular_frequency, time, phase, math.sin)
    omega_rad_s = float(omega.to("rad / s").magnitude)
    vel = -amp_m * omega_rad_s * sin_phase
    return Quantity(0.0 if vel == 0 else vel, "m / s")


def shm_acceleration(
    amplitude: QuantityLike,
    angular_frequency: QuantityLike,
    time: QuantityLike,
    *,
    phase: QuantityLike = "0 rad",
) -> pint.Quantity:
    """Compute SHM acceleration: ``a(t) = -A * omega^2 * cos(omega * t + phase)``.

    Args:
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).
        time: Elapsed time (dimension ``[time]``, ``>= 0``).
        phase: Initial phase constant (compatible with ``"rad"``, defaults to ``"0 rad"``).

    Returns:
        Instantaneous acceleration as a ``pint.Quantity`` in ``meter / second ** 2``.
    """
    amp_m = float(
        validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
        .to("m")
        .magnitude
    )
    omega, cos_phase = _eval_phase_trig(angular_frequency, time, phase, math.cos)
    omega_rad_s = float(omega.to("rad / s").magnitude)
    acc = -amp_m * (omega_rad_s**2) * cos_phase
    return Quantity(0.0 if acc == 0 else acc, "m / s^2")


def shm_max_speed(
    amplitude: QuantityLike,
    angular_frequency: QuantityLike,
) -> pint.Quantity:
    """Compute maximum SHM speed: ``v_max = A * |omega|``.

    Args:
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).

    Returns:
        Maximum speed as a ``pint.Quantity`` in ``meter / second``.
    """
    amp_m = float(
        validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
        .to("m")
        .magnitude
    )
    omega_rad_s = abs(
        float(
            validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
            .to("rad / s")
            .magnitude
        )
    )
    return Quantity(amp_m * omega_rad_s, "m / s")


def shm_max_acceleration(
    amplitude: QuantityLike,
    angular_frequency: QuantityLike,
) -> pint.Quantity:
    """Compute maximum SHM acceleration magnitude: ``a_max = A * omega^2``.

    Args:
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).

    Returns:
        Maximum acceleration magnitude as a ``pint.Quantity`` in ``meter / second ** 2``.
    """
    amp_m = float(
        validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
        .to("m")
        .magnitude
    )
    omega_rad_s = float(
        validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
        .to("rad / s")
        .magnitude
    )
    return Quantity(amp_m * (omega_rad_s**2), "m / s^2")


def shm_energy(
    spring_constant: QuantityLike,
    amplitude: QuantityLike,
) -> pint.Quantity:
    """Compute total mechanical energy of a harmonic oscillator: ``E = 0.5 * k * A^2``.

    Args:
        spring_constant: Effective spring stiffness (compatible with ``N / m``, ``>= 0``).
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).

    Returns:
        Total mechanical energy as a ``pint.Quantity`` in ``joule``.
    """
    amp = validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
    return elastic_potential_energy(spring_constant, amp)


def shm_speed_at_position(
    angular_frequency: QuantityLike,
    amplitude: QuantityLike,
    displacement: QuantityLike,
) -> pint.Quantity:
    """Compute SHM speed magnitude at displacement ``x``: ``v = |omega| * sqrt(A^2 - x^2)``.

    Args:
        angular_frequency: Angular frequency (compatible with ``"rad / s"``).
        amplitude: Oscillation amplitude (dimension ``[length]``, ``>= 0``).
        displacement: Instantaneous displacement (dimension ``[length]``,
            satisfying ``|x| <= amplitude``).

    Returns:
        Speed magnitude as a ``pint.Quantity`` in ``meter / second``.

    Raises:
        PhysicalDomainError: If ``|displacement| > amplitude``.
    """
    omega_rad_s = abs(
        float(
            validate_quantity(angular_frequency, "rad / s", name="angular_frequency")
            .to("rad / s")
            .magnitude
        )
    )
    amp_m = float(
        validate_quantity(amplitude, "[length]", name="amplitude", allow_negative=False)
        .to("m")
        .magnitude
    )
    x_m = float(validate_quantity(displacement, "[length]", name="displacement").to("m").magnitude)
    if abs(x_m) > amp_m:
        raise PhysicalDomainError(
            f"Displacement magnitude ({abs(x_m)} m) cannot exceed amplitude ({amp_m} m)."
        )
    diff_sq = max(0.0, (amp_m**2) - (x_m**2))
    return Quantity(omega_rad_s * math.sqrt(diff_sq), "m / s")


__all__ = [
    "physical_pendulum_period",
    "shm_acceleration",
    "shm_displacement",
    "shm_energy",
    "shm_max_acceleration",
    "shm_max_speed",
    "shm_speed_at_position",
    "shm_velocity",
    "simple_pendulum_period",
    "spring_mass_angular_frequency",
    "spring_mass_period",
]
