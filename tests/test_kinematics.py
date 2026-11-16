"""Unit tests for physu.kinematics submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError, ureg
from physu.kinematics import (
    accelerated,
    accelerated_displacement,
    accelerated_position,
    acceleration,
    angular_displacement,
    angular_velocity_from_frequency,
    angular_velocity_from_period,
    angular_velocity_from_speed,
    arc_length,
    average_velocity,
    centripetal_acceleration,
    displacement,
    displacement_from_velocities,
    final_velocity,
    free_fall_distance,
    free_fall_speed,
    free_fall_time,
    frequency_from_angular_velocity,
    frequency_from_period,
    max_height,
    period_from_angular_velocity,
    period_from_frequency,
    position,
    projectile_flight_time,
    projectile_max_height,
    projectile_position,
    projectile_range,
    projectile_velocity,
    tangential_speed,
    time_to_displacement,
    time_to_height,
    time_to_peak,
    time_to_velocity,
    torricelli_velocity,
    travel_time,
    velocity_components,
    vertical_position,
    vertical_velocity,
)

# --- Uniform Rectilinear Motion (MRU) ---


def test_uniform_motion_calculations_and_conversions() -> None:
    dx = displacement("72 km/h", "10 s")
    assert dx.to("m").magnitude == pytest.approx(200.0)
    assert dx.to("km").magnitude == pytest.approx(0.2)

    x = position(100 * ureg.meter, 10 * ureg.meter / ureg.second, "5 s")
    assert x.to("m").magnitude == pytest.approx(150.0)

    v = average_velocity("0.15 km", "15 s")
    assert v.to("km / h").magnitude == pytest.approx(36.0)

    t = travel_time("1 km", "20 m/s")
    assert t.to("s").magnitude == pytest.approx(50.0)


def test_uniform_motion_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        displacement("10 kg", "2 s")
    with pytest.raises(PhysicalDomainError):
        displacement("10 m/s", "-1 s")
    with pytest.raises(PhysicalDomainError):
        average_velocity("10 m", "0 s")
    with pytest.raises(PhysicalDomainError):
        travel_time("10 m", "0 m/s")
    with pytest.raises(PhysicalDomainError):
        travel_time("10 m", "-2 m/s")


# --- Uniformly Accelerated Motion (MRUA) ---


def test_accelerated_motion_calculations() -> None:
    assert final_velocity("0 m/s", "2 m/s^2", "5 s").to("m/s").magnitude == pytest.approx(10.0)
    assert accelerated_displacement("36 km/h", "2 m/s^2", "5 s").to("m").magnitude == pytest.approx(
        75.0
    )
    assert accelerated.displacement("0 m/s", "2 m/s^2", "4 s").magnitude == pytest.approx(16.0)
    assert accelerated_position(
        "10 m", "0 m/s", "2 m/s^2", "3 s"
    ).to("m").magnitude == pytest.approx(19.0)
    assert displacement_from_velocities(
        "0 m/s", "20 m/s", "4 s"
    ).to("m").magnitude == pytest.approx(40.0)
    assert acceleration("0 m/s", "36 km/h", "5 s").to("m/s^2").magnitude == pytest.approx(2.0)
    assert torricelli_velocity("0 m/s", "2 m/s^2", "25 m").to("m/s").magnitude == pytest.approx(
        10.0
    )
    assert time_to_velocity("0 m/s", "10 m/s", "2 m/s^2").to("s").magnitude == pytest.approx(5.0)


def test_accelerated_time_to_displacement_cases() -> None:
    roots_two = time_to_displacement("20 m/s", "-10 m/s^2", "15 m")
    assert [r.to("s").magnitude for r in roots_two] == pytest.approx([1.0, 3.0])

    roots_one = time_to_displacement("20 m/s", "-10 m/s^2", "20 m")
    assert [r.to("s").magnitude for r in roots_one] == pytest.approx([2.0])

    roots_linear = time_to_displacement("10 m/s", "0 m/s^2", "50 m")
    assert [r.to("s").magnitude for r in roots_linear] == pytest.approx([5.0])


def test_accelerated_motion_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        final_velocity("10 m", "2 m/s^2", "1 s")
    with pytest.raises(PhysicalDomainError):
        final_velocity("0 m/s", "2 m/s^2", "-1 s")
    with pytest.raises(PhysicalDomainError):
        acceleration("0 m/s", "10 m/s", "0 s")
    with pytest.raises(PhysicalDomainError):
        torricelli_velocity("0 m/s", "-2 m/s^2", "10 m")
    with pytest.raises(PhysicalDomainError):
        time_to_velocity("10 m/s", "20 m/s", "-2 m/s^2")
    with pytest.raises(PhysicalDomainError):
        time_to_displacement("10 m/s", "-10 m/s^2", "10 m")
    with pytest.raises(PhysicalDomainError):
        time_to_displacement("0 m/s", "0 m/s^2", "0 m")
    with pytest.raises(PhysicalDomainError):
        time_to_displacement("0 m/s", "0 m/s^2", "5 m")


# --- Vertical Motion & Free Fall ---


def test_vertical_motion_calculations() -> None:
    assert vertical_position("0 m", "20 m/s", "2 s", gravity="10 m/s^2").to(
        "m"
    ).magnitude == pytest.approx(20.0)
    assert vertical_velocity("20 m/s", "3 s", gravity="10 m/s^2").to(
        "m/s"
    ).magnitude == pytest.approx(-10.0)
    assert free_fall_distance("3 s", gravity="10 m/s^2").to("m").magnitude == pytest.approx(45.0)
    assert free_fall_distance("1 s").to("m").magnitude == pytest.approx(0.5 * 9.80665)
    assert free_fall_speed("20 m", gravity="10 m/s^2").to("m/s").magnitude == pytest.approx(20.0)
    assert free_fall_time("20 m", gravity="10 m/s^2").to("s").magnitude == pytest.approx(2.0)
    assert time_to_peak("20 m/s", gravity="10 m/s^2").to("s").magnitude == pytest.approx(2.0)
    assert max_height("20 m/s", initial_height="5 m", gravity="10 m/s^2").to(
        "m"
    ).magnitude == pytest.approx(25.0)

    times = time_to_height("20 m/s", "15 m", gravity="10 m/s^2")
    assert [t.to("s").magnitude for t in times] == pytest.approx([1.0, 3.0])


def test_vertical_motion_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        free_fall_time("5 s")
    with pytest.raises(PhysicalDomainError):
        free_fall_distance("1 s", gravity="-9.8 m/s^2")
    with pytest.raises(PhysicalDomainError):
        free_fall_time("10 m", gravity="0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        free_fall_speed("-5 m")
    with pytest.raises(PhysicalDomainError):
        max_height("-5 m/s")
    with pytest.raises(PhysicalDomainError):
        time_to_height("20 m/s", "25 m", gravity="10 m/s^2")


# --- 2D Projectile Motion ---


def test_projectile_motion_calculations() -> None:
    vx0, vy0 = velocity_components("20 m/s", "30 deg")
    assert vx0.to("m/s").magnitude == pytest.approx(10.0 * math.sqrt(3.0))
    assert vy0.to("m/s").magnitude == pytest.approx(10.0)

    vx_vert, vy_vert = velocity_components("20 m/s", "90 deg")
    assert vx_vert.magnitude == 0.0
    assert vy_vert.magnitude == pytest.approx(20.0)

    assert projectile_flight_time("20 m/s", "45 deg", gravity="10 m/s^2").to(
        "s"
    ).magnitude == pytest.approx(2.0 * math.sqrt(2.0))
    assert projectile_max_height("20 m/s", "30 deg", gravity="10 m/s^2").to(
        "m"
    ).magnitude == pytest.approx(5.0)
    assert projectile_range("20 m/s", "45 deg", gravity="10 m/s^2").to(
        "m"
    ).magnitude == pytest.approx(40.0)
    assert projectile_range("20 m/s", "90 deg", gravity="10 m/s^2").magnitude == 0.0

    # Horizontal launch from an elevated cliff
    assert projectile_flight_time(
        "10 m/s", "0 deg", initial_height="20 m", gravity="10 m/s^2"
    ).to("s").magnitude == pytest.approx(2.0)
    assert projectile_range(
        "10 m/s", "0 deg", initial_height="20 m", gravity="10 m/s^2"
    ).to("m").magnitude == pytest.approx(20.0)

    x, y = projectile_position("20 m/s", "30 deg", "1 s", gravity="10 m/s^2")
    assert x.to("m").magnitude == pytest.approx(10.0 * math.sqrt(3.0))
    assert y.to("m").magnitude == pytest.approx(5.0)

    vx, vy = projectile_velocity("20 m/s", "30 deg", "1 s", gravity="10 m/s^2")
    assert vx.to("m/s").magnitude == pytest.approx(10.0 * math.sqrt(3.0))
    assert vy.to("m/s").magnitude == pytest.approx(0.0, abs=1e-12)


def test_projectile_motion_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        velocity_components("10 m/s", "5 m")
    with pytest.raises(PhysicalDomainError):
        velocity_components("-5 m/s", "30 deg")
    with pytest.raises(PhysicalDomainError):
        projectile_range("10 m/s", "30 deg", initial_height="-2 m")
    with pytest.raises(PhysicalDomainError):
        projectile_flight_time("10 m/s", "-30 deg")
    with pytest.raises(PhysicalDomainError):
        projectile_max_height("10 m/s", "-30 deg")
    with pytest.raises(PhysicalDomainError):
        projectile_velocity("10 m/s", "30 deg", "-1 s")


# --- Uniform Circular Motion (UCM) ---


def test_circular_motion_calculations() -> None:
    assert tangential_speed("10 rad/s", "2 m").to("m/s").magnitude == pytest.approx(20.0)
    assert tangential_speed("60 rpm", "1 m").to("m/s").magnitude == pytest.approx(2.0 * math.pi)
    assert angular_velocity_from_speed("20 m/s", "2 m").to(
        "rad/s"
    ).magnitude == pytest.approx(10.0)

    assert centripetal_acceleration("2 m", tangential_speed="10 m/s").to(
        "m/s^2"
    ).magnitude == pytest.approx(50.0)
    assert centripetal_acceleration("2 m", angular_velocity="5 rad/s").to(
        "m/s^2"
    ).magnitude == pytest.approx(50.0)

    assert period_from_frequency("2 Hz").to("s").magnitude == pytest.approx(0.5)
    assert frequency_from_period("0.5 s").to("Hz").magnitude == pytest.approx(2.0)
    assert angular_velocity_from_period("2 s").to("rad/s").magnitude == pytest.approx(math.pi)
    assert angular_velocity_from_frequency("2 Hz").to("rad/s").magnitude == pytest.approx(
        4.0 * math.pi
    )
    assert period_from_angular_velocity("60 rpm").to("s").magnitude == pytest.approx(1.0)
    assert frequency_from_angular_velocity("60 rpm").to("Hz").magnitude == pytest.approx(1.0)
    assert angular_displacement("10 rad/s", "3 s").to("rad").magnitude == pytest.approx(30.0)
    assert arc_length("2 m", "180 deg").to("m").magnitude == pytest.approx(2.0 * math.pi)


def test_circular_motion_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        arc_length("2 s", "1 rad")
    with pytest.raises(PhysicalDomainError):
        tangential_speed("10 rad/s", "0 m")
    with pytest.raises(PhysicalDomainError):
        centripetal_acceleration("-2 m", tangential_speed="5 m/s")
    with pytest.raises(ValueError, match="exactly one"):
        centripetal_acceleration("2 m")
    with pytest.raises(ValueError, match="exactly one"):
        centripetal_acceleration("2 m", tangential_speed="5 m/s", angular_velocity="2 rad/s")
    with pytest.raises(PhysicalDomainError):
        period_from_frequency("0 Hz")
    with pytest.raises(PhysicalDomainError):
        period_from_angular_velocity("0 rad/s")
