"""Magnetism formulas: Lorentz force, cyclotron motion, and magnetic fields."""

from __future__ import annotations

import math

import pint

from ..constants import VACUUM_PERMEABILITY
from ..units import QuantityLike
from ..validation import validate_quantity
from ._helpers import (
    _sin_angle,
    _to_clean_unit,
    _validate_positive_factor,
    _validate_turns,
)


def lorentz_magnetic_force(
    charge: QuantityLike,
    speed: QuantityLike,
    magnetic_field: QuantityLike,
    *,
    angle: QuantityLike = "90 deg",
) -> pint.Quantity:
    """Compute magnetic Lorentz force magnitude: ``F_B = |q| * v * B * |sin(theta)|``.

    Args:
        charge: Electric charge (dimension ``[charge]``).
        speed: Particle speed (dimension ``[velocity]``, ``>= 0``).
        magnetic_field: Magnetic flux density (dimension ``[magnetic_field]``, ``>= 0``).
        angle: Angle between velocity and magnetic field vectors (compatible with
            ``"rad"``, defaults to ``"90 deg"``).

    Returns:
        Magnetic force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    b = validate_quantity(
        magnetic_field,
        "[magnetic_field]",
        name="magnetic_field",
        allow_negative=False,
    )
    sin_theta = abs(_sin_angle(angle))
    return _to_clean_unit(abs(q) * v * b * sin_theta, "N")


def magnetic_force_on_wire(
    current: QuantityLike,
    length: QuantityLike,
    magnetic_field: QuantityLike,
    *,
    angle: QuantityLike = "90 deg",
) -> pint.Quantity:
    """Compute magnetic force magnitude on a straight conductor: ``F = |I| * L * B * |sin(theta)|``.

    Args:
        current: Electric current (dimension ``[current]``).
        length: Conductor length in the magnetic field (dimension ``[length]``, ``>= 0``).
        magnetic_field: Magnetic flux density (dimension ``[magnetic_field]``, ``>= 0``).
        angle: Angle between conductor and magnetic field (compatible with
            ``"rad"``, defaults to ``"90 deg"``).

    Returns:
        Magnetic force magnitude as a ``pint.Quantity`` in ``newton``.
    """
    i = validate_quantity(current, "[current]", name="current")
    length_q = validate_quantity(length, "[length]", name="length", allow_negative=False)
    b = validate_quantity(
        magnetic_field,
        "[magnetic_field]",
        name="magnetic_field",
        allow_negative=False,
    )
    sin_theta = abs(_sin_angle(angle))
    return _to_clean_unit(abs(i) * length_q * b * sin_theta, "N")


def cyclotron_radius(
    mass: QuantityLike,
    speed: QuantityLike,
    charge: QuantityLike,
    magnetic_field: QuantityLike,
) -> pint.Quantity:
    """Compute cyclotron orbit radius of a charged particle: ``r = m * v / (|q| * B)``.

    Args:
        mass: Particle mass (dimension ``[mass]``, ``>= 0``).
        speed: Perpendicular particle speed (dimension ``[velocity]``, ``>= 0``).
        charge: Non-zero particle charge (dimension ``[charge]``, ``!= 0``).
        magnetic_field: Magnetic flux density (dimension ``[magnetic_field]``, ``> 0``).

    Returns:
        Cyclotron radius as a ``pint.Quantity`` in ``meter``.
    """
    m = validate_quantity(mass, "[mass]", name="mass", allow_negative=False)
    v = validate_quantity(speed, "[velocity]", name="speed", allow_negative=False)
    q = validate_quantity(charge, "[charge]", name="charge", allow_zero=False)
    b = validate_quantity(
        magnetic_field,
        "[magnetic_field]",
        name="magnetic_field",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit((m * v) / (abs(q) * b), "m")


def cyclotron_frequency(
    charge: QuantityLike,
    magnetic_field: QuantityLike,
    mass: QuantityLike,
) -> pint.Quantity:
    """Compute cyclotron frequency: ``f_c = |q| * B / (2 * pi * m)``.

    Args:
        charge: Particle charge (dimension ``[charge]``).
        magnetic_field: Magnetic flux density (dimension ``[magnetic_field]``, ``>= 0``).
        mass: Positive particle mass (dimension ``[mass]``, ``> 0``).

    Returns:
        Cyclotron frequency as a ``pint.Quantity`` in ``hertz``.
    """
    q = validate_quantity(charge, "[charge]", name="charge")
    b = validate_quantity(
        magnetic_field,
        "[magnetic_field]",
        name="magnetic_field",
        allow_negative=False,
    )
    m = validate_quantity(
        mass,
        "[mass]",
        name="mass",
        allow_negative=False,
        allow_zero=False,
    )
    return _to_clean_unit((abs(q) * b) / (2.0 * math.pi * m), "Hz")


def magnetic_field_wire(
    current: QuantityLike,
    distance: QuantityLike,
    *,
    relative_permeability: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute long straight wire magnetic field: ``B = mu_0 * mu_r * |I| / (2 * pi * r)``.

    Args:
        current: Electric current in the wire (dimension ``[current]``).
        distance: Radial distance from wire axis (dimension ``[length]``, ``> 0``).
        relative_permeability: Medium relative permeability (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Magnetic flux density magnitude as a ``pint.Quantity`` in ``tesla``.
    """
    i = validate_quantity(current, "[current]", name="current")
    r = validate_quantity(
        distance,
        "[length]",
        name="distance",
        allow_negative=False,
        allow_zero=False,
    )
    mu_r = _validate_positive_factor(
        relative_permeability,
        name="relative_permeability",
    )
    return _to_clean_unit(
        (VACUUM_PERMEABILITY * mu_r * abs(i)) / (2.0 * math.pi * r),
        "T",
    )


def magnetic_field_loop_center(
    current: QuantityLike,
    radius: QuantityLike,
    *,
    turns: QuantityLike = 1,
    relative_permeability: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute circular loop center magnetic field: ``B = mu_0 * mu_r * N * |I| / (2 * R)``.

    Args:
        current: Electric current in the coil (dimension ``[current]``).
        radius: Loop radius (dimension ``[length]``, ``> 0``).
        turns: Number of wire turns (dimensionless, ``> 0``, defaults to ``1``).
        relative_permeability: Core relative permeability (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Magnetic flux density magnitude as a ``pint.Quantity`` in ``tesla``.
    """
    i = validate_quantity(current, "[current]", name="current")
    r = validate_quantity(
        radius,
        "[length]",
        name="radius",
        allow_negative=False,
        allow_zero=False,
    )
    n_turns = _validate_turns(turns)
    mu_r = _validate_positive_factor(
        relative_permeability,
        name="relative_permeability",
    )
    return _to_clean_unit(
        (VACUUM_PERMEABILITY * mu_r * n_turns * abs(i)) / (2.0 * r),
        "T",
    )


def magnetic_field_solenoid(
    turns: QuantityLike,
    length: QuantityLike,
    current: QuantityLike,
    *,
    relative_permeability: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute ideal solenoid magnetic field: ``B = mu_0 * mu_r * (N / L) * |I|``.

    Args:
        turns: Total number of turns (dimensionless, ``> 0``).
        length: Solenoid length (dimension ``[length]``, ``> 0``).
        current: Winding electric current (dimension ``[current]``).
        relative_permeability: Core relative permeability (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Magnetic flux density magnitude as a ``pint.Quantity`` in ``tesla``.
    """
    n_turns = _validate_turns(turns)
    length_q = validate_quantity(
        length,
        "[length]",
        name="length",
        allow_negative=False,
        allow_zero=False,
    )
    i = validate_quantity(current, "[current]", name="current")
    mu_r = _validate_positive_factor(
        relative_permeability,
        name="relative_permeability",
    )
    return _to_clean_unit(
        VACUUM_PERMEABILITY * mu_r * (n_turns / length_q) * abs(i),
        "T",
    )


def force_per_length_parallel_wires(
    current_1: QuantityLike,
    current_2: QuantityLike,
    separation: QuantityLike,
    *,
    relative_permeability: QuantityLike = 1.0,
) -> pint.Quantity:
    """Compute parallel-wire force per length: ``F / L = mu_0 * mu_r * |I1 * I2| / (2 * pi * d)``.

    Args:
        current_1: Current in the first wire (dimension ``[current]``).
        current_2: Current in the second wire (dimension ``[current]``).
        separation: Center-to-center wire separation (dimension ``[length]``, ``> 0``).
        relative_permeability: Medium relative permeability (dimensionless, ``> 0``,
            defaults to ``1.0``).

    Returns:
        Force per unit length as a ``pint.Quantity`` in ``newton / meter``.
    """
    sep = validate_quantity(
        separation,
        "[length]",
        name="separation",
        allow_negative=False,
        allow_zero=False,
    )
    i2 = validate_quantity(current_2, "[current]", name="current_2")
    b_from_1 = magnetic_field_wire(
        current_1,
        sep,
        relative_permeability=relative_permeability,
    )
    return _to_clean_unit(b_from_1 * abs(i2), "N / m")


__all__ = [
    "cyclotron_frequency",
    "cyclotron_radius",
    "force_per_length_parallel_wires",
    "lorentz_magnetic_force",
    "magnetic_field_loop_center",
    "magnetic_field_solenoid",
    "magnetic_field_wire",
    "magnetic_force_on_wire",
]
