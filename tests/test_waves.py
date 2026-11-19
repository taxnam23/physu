"""Unit tests for physu.waves submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import STANDARD_GRAVITY
from physu.waves import (
    physical_pendulum_period,
    shm_acceleration,
    shm_displacement,
    shm_energy,
    shm_max_acceleration,
    shm_max_speed,
    shm_speed_at_position,
    shm_velocity,
    simple_pendulum_period,
    spring_mass_angular_frequency,
    spring_mass_period,
)


def test_spring_mass_and_pendulums() -> None:
    omega = spring_mass_angular_frequency("400 N / m", "1 kg")
    assert str(omega.units) == "radian / second"
    assert omega.magnitude == pytest.approx(20.0)

    t_spring = spring_mass_period("500 g", "200 N / m")
    assert str(t_spring.units) == "second"
    assert t_spring.magnitude == pytest.approx(2.0 * math.pi * math.sqrt(0.5 / 200.0))

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    t_pend = simple_pendulum_period("100 cm")
    assert str(t_pend.units) == "second"
    assert t_pend.magnitude == pytest.approx(2.0 * math.pi * math.sqrt(1.0 / g_std))

    # Uniform rod of mass 1.5 kg, length 1.2 m pivoted at one end: I = 1/3 m L^2, d = L/2
    inertia = (1.0 / 3.0) * 1.5 * (1.2**2)
    t_phys = physical_pendulum_period(
        f"{inertia} kg * m^2",
        "1.5 kg",
        "0.6 m",
        gravity="9.81 m/s^2",
    )
    assert str(t_phys.units) == "second"
    assert t_phys.magnitude == pytest.approx(2.0 * math.pi * math.sqrt((2.0 * 1.2) / (3.0 * 9.81)))


def test_shm_kinematics_and_energy() -> None:
    # At t = 0 with phase = 90 deg: x = 0, v = -A * omega, a = 0
    x_90 = shm_displacement("10 cm", "4 rad / s", "0 s", phase="90 deg")
    v_90 = shm_velocity("10 cm", "4 rad / s", "0 s", phase="90 deg")
    a_90 = shm_acceleration("10 cm", "4 rad / s", "0 s", phase="90 deg")
    assert str(x_90.units) == "meter"
    assert x_90.magnitude == 0.0
    assert str(v_90.units) == "meter / second"
    assert v_90.magnitude == pytest.approx(-0.4)
    assert str(a_90.units) == "meter / second ** 2"
    assert a_90.magnitude == 0.0

    # At t = 0 with phase = 0: x = A, v = 0, a = -A * omega^2
    assert shm_displacement("10 cm", "4 rad / s", "0 s").to("cm").magnitude == pytest.approx(10.0)
    assert shm_velocity("10 cm", "4 rad / s", "0 s").magnitude == 0.0
    assert shm_acceleration("10 cm", "4 rad / s", "0 s").magnitude == pytest.approx(-1.6)

    assert shm_max_speed("10 cm", "4 rad / s").magnitude == pytest.approx(0.4)
    assert shm_max_acceleration("10 cm", "4 rad / s").magnitude == pytest.approx(1.6)
    assert shm_energy("200 N / m", "5 cm").magnitude == pytest.approx(0.25)

    v_mid = shm_speed_at_position("5 rad / s", "10 cm", "-6 cm")
    assert v_mid.magnitude == pytest.approx(5.0 * 0.08)
    assert shm_speed_at_position("5 rad / s", "10 cm", "10 cm").magnitude == 0.0


def test_oscillations_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        spring_mass_angular_frequency("100 N", "1 kg")
    with pytest.raises(pint.DimensionalityError):
        simple_pendulum_period("1 s")
    with pytest.raises(pint.DimensionalityError):
        physical_pendulum_period("1 kg * m", "1 kg", "0.5 m")
    with pytest.raises(pint.DimensionalityError):
        shm_displacement("10 cm", "4 m / s", "1 s")
    with pytest.raises(pint.DimensionalityError):
        shm_displacement("10 cm", "4 rad / s", "1 s", phase="1 m")

    with pytest.raises(PhysicalDomainError):
        spring_mass_angular_frequency("-100 N / m", "1 kg")
    with pytest.raises(PhysicalDomainError):
        spring_mass_angular_frequency("100 N / m", "0 kg")
    with pytest.raises(PhysicalDomainError):
        spring_mass_period("1 kg", "0 N / m")
    with pytest.raises(PhysicalDomainError):
        simple_pendulum_period("-1 m")
    with pytest.raises(PhysicalDomainError):
        simple_pendulum_period("1 m", gravity="0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        physical_pendulum_period("1 kg * m^2", "1 kg", "0 m")
    with pytest.raises(PhysicalDomainError):
        shm_displacement("-5 cm", "4 rad / s", "1 s")
    with pytest.raises(PhysicalDomainError):
        shm_velocity("5 cm", "4 rad / s", "-1 s")
    with pytest.raises(PhysicalDomainError):
        shm_energy("100 N / m", "-2 cm")
    with pytest.raises(PhysicalDomainError):
        shm_speed_at_position("4 rad / s", "10 cm", "12 cm")
