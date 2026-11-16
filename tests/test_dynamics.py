"""Unit tests for physu.dynamics submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError, ureg
from physu.constants import GRAVITATIONAL_CONSTANT
from physu.dynamics import (
    acceleration,
    average_force,
    centripetal_force,
    collision_velocities,
    critical_incline_angle,
    elastic_collision_velocities,
    force,
    friction_coefficient,
    gravitational_field_strength,
    gravitational_force,
    impulse,
    inclined_normal_force,
    inclined_parallel_force,
    inelastic_collision_velocity,
    kinetic_friction,
    linear_momentum,
    mass,
    max_static_friction,
    momentum_change,
    normal_force,
    orbital_speed,
    parallel_spring_constant,
    series_spring_constant,
    spring_constant,
    spring_displacement,
    spring_force,
    spring_force_magnitude,
    weight,
)


def test_newton_second_law_and_unit_conversions() -> None:
    f = force("2000 g", "3 m/s^2")
    assert str(f.units) == "newton"
    assert f.magnitude == pytest.approx(6.0)

    a = acceleration("10 N", 2 * ureg.kilogram)
    assert a.to("m / s^2").magnitude == pytest.approx(5.0)

    m = mass("20 N", "4 m/s^2")
    assert m.to("kg").magnitude == pytest.approx(5.0)

    f_lbf = force("10 kg", "5 m/s^2")
    assert f_lbf.to("lbf").magnitude == pytest.approx(11.2404471549856, rel=1e-6)


def test_weight_and_inclined_plane_forces() -> None:
    w_default = weight("10 kg")
    assert w_default.to("N").magnitude == pytest.approx(98.0665)

    w_custom = weight("10 kg", gravity="10 m/s^2")
    assert w_custom.to("N").magnitude == pytest.approx(100.0)

    f_par = inclined_parallel_force("10 kg", "30 deg", gravity="10 m/s^2")
    assert f_par.to("N").magnitude == pytest.approx(50.0)

    n_force = inclined_normal_force("10 kg", "60 deg", gravity="10 m/s^2")
    assert n_force.to("N").magnitude == pytest.approx(50.0)

    assert inclined_parallel_force("10 kg", "0 deg").magnitude == 0.0
    assert inclined_normal_force("10 kg", "90 deg").magnitude == 0.0


def test_newton_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        force("10 m", "2 m/s^2")
    with pytest.raises(pint.DimensionalityError):
        inclined_parallel_force("10 kg", "5 m")

    with pytest.raises(PhysicalDomainError):
        force("-2 kg", "3 m/s^2")
    with pytest.raises(PhysicalDomainError):
        acceleration("10 N", "0 kg")
    with pytest.raises(PhysicalDomainError):
        mass("10 N", "0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        mass("10 N", "-2 m/s^2")
    with pytest.raises(PhysicalDomainError):
        weight("10 kg", gravity="-9.8 m/s^2")
    with pytest.raises(PhysicalDomainError):
        inclined_normal_force("10 kg", "-10 deg")
    with pytest.raises(PhysicalDomainError):
        inclined_normal_force("10 kg", "100 deg")


def test_friction_calculations_and_conversions() -> None:
    fs_max = max_static_friction(0.5, "100 N")
    assert str(fs_max.units) == "newton"
    assert fs_max.magnitude == pytest.approx(50.0)

    fk = kinetic_friction(0.3, "200 N")
    assert fk.to("N").magnitude == pytest.approx(60.0)

    mu = friction_coefficient("30 N", "150 N")
    assert mu.dimensionless
    assert mu.magnitude == pytest.approx(0.2)

    n_flat = normal_force("10 kg", gravity="10 m/s^2")
    assert n_flat.to("N").magnitude == pytest.approx(100.0)

    n_incline = normal_force("10 kg", angle="60 deg", gravity="10 m/s^2")
    assert n_incline.to("N").magnitude == pytest.approx(50.0)

    theta_c = critical_incline_angle(1.0)
    assert theta_c.to("deg").magnitude == pytest.approx(45.0)


def test_friction_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        max_static_friction("0.4 m", "100 N")
    with pytest.raises(pint.DimensionalityError):
        kinetic_friction(0.3, "10 kg")
    with pytest.raises(pint.DimensionalityError):
        critical_incline_angle("1 s")

    with pytest.raises(PhysicalDomainError):
        max_static_friction(-0.1, "100 N")
    with pytest.raises(PhysicalDomainError):
        kinetic_friction(0.3, "-10 N")
    with pytest.raises(PhysicalDomainError):
        friction_coefficient("10 N", "0 N")
    with pytest.raises(PhysicalDomainError):
        friction_coefficient("-5 N", "50 N")
    with pytest.raises(PhysicalDomainError):
        critical_incline_angle(-0.5)


def test_elasticity_calculations_and_conversions() -> None:
    f_restoring = spring_force("200 N/m", "15 cm")
    assert str(f_restoring.units) == "newton"
    assert f_restoring.magnitude == pytest.approx(-30.0)

    f_zero = spring_force("200 N/m", "0 m")
    assert f_zero.magnitude == 0.0
    assert str(f_zero.magnitude) == "0.0"

    f_mag = spring_force_magnitude("2 kN/m", "-5 cm")
    assert f_mag.to("N").magnitude == pytest.approx(100.0)

    k_from_restoring = spring_constant("-30 N", "15 cm")
    assert str(k_from_restoring.units) == "newton / meter"
    assert k_from_restoring.magnitude == pytest.approx(200.0)

    k_from_applied = spring_constant("30 N", "15 cm")
    assert k_from_applied.to("N / m").magnitude == pytest.approx(200.0)

    dx = spring_displacement("-30 N", "200 N/m")
    assert dx.to("cm").magnitude == pytest.approx(15.0)
    assert spring_displacement("0 N", "200 N/m").magnitude == 0.0

    k_series = series_spring_constant(["100 N/m", "200 N/m", "2 N/cm"])
    assert k_series.to("N / m").magnitude == pytest.approx(50.0)

    k_parallel = parallel_spring_constant(["100 N/m", "1 N/cm", "0.3 kN/m"])
    assert k_parallel.to("N / m").magnitude == pytest.approx(500.0)


def test_elasticity_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        spring_force("200 N", "0.1 m")
    with pytest.raises(pint.DimensionalityError):
        spring_force("200 N/m", "2 s")
    with pytest.raises(pint.DimensionalityError):
        spring_constant("10 kg", "0.2 m")
    with pytest.raises(pint.DimensionalityError):
        series_spring_constant(["100 N/m", "50 N"])

    with pytest.raises(TypeError):
        series_spring_constant("100 N/m")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parallel_spring_constant("100 N/m")  # type: ignore[arg-type]

    with pytest.raises(PhysicalDomainError):
        spring_force("-10 N/m", "0.1 m")
    with pytest.raises(PhysicalDomainError):
        spring_force_magnitude("-10 N/m", "0.1 m")
    with pytest.raises(PhysicalDomainError):
        spring_constant("10 N", "0 m")
    with pytest.raises(PhysicalDomainError):
        spring_displacement("10 N", "0 N/m")
    with pytest.raises(PhysicalDomainError):
        spring_displacement("10 N", "-50 N/m")
    with pytest.raises(PhysicalDomainError):
        series_spring_constant([])
    with pytest.raises(PhysicalDomainError):
        series_spring_constant(["100 N/m", "0 N/m"])
    with pytest.raises(PhysicalDomainError):
        parallel_spring_constant([])
    with pytest.raises(PhysicalDomainError):
        parallel_spring_constant(["100 N/m", "-20 N/m"])


def test_momentum_and_impulse_calculations() -> None:
    p = linear_momentum("2000 g", "72 km/h")
    assert p.to("kg * m / s").magnitude == pytest.approx(40.0)
    assert p.to("N * s").magnitude == pytest.approx(40.0)

    j = impulse("50 N", "400 ms")
    assert j.to("N * s").magnitude == pytest.approx(20.0)

    dp = momentum_change("2 kg", "-3 m/s", "5 m/s")
    assert dp.to("kg * m / s").magnitude == pytest.approx(16.0)

    f_avg = average_force("20 N * s", "0.5 s")
    assert str(f_avg.units) == "newton"
    assert f_avg.magnitude == pytest.approx(40.0)


def test_one_dimensional_collisions() -> None:
    # Equal-mass elastic collision swaps velocities
    v1_eq, v2_eq = elastic_collision_velocities("2 kg", "5 m/s", "2000 g", "-1 m/s")
    assert v1_eq.to("m/s").magnitude == pytest.approx(-1.0)
    assert v2_eq.to("m/s").magnitude == pytest.approx(5.0)

    # Unequal masses: v1' = ((1 - 3)*4 + 2*3*0)/4 = -2 m/s, v2' = (2*1*4)/4 = 2 m/s
    v1_el, v2_el = elastic_collision_velocities("1 kg", "4 m/s", "3 kg", "0 m/s")
    assert v1_el.to("m/s").magnitude == pytest.approx(-2.0)
    assert v2_el.to("m/s").magnitude == pytest.approx(2.0)

    # Perfectly inelastic collision
    v_inel = inelastic_collision_velocity("2 kg", "6 m/s", "4 kg", "-3 m/s")
    assert v_inel.to("m/s").magnitude == pytest.approx(0.0)

    # Partial restitution (e = 0.5): e*(v1 - v2) = 0.5*(6 - 0) = 3 -> v2' - v1' = 3
    v1_part, v2_part = collision_velocities("2 kg", "6 m/s", "2 kg", "0 m/s", restitution=0.5)
    assert v1_part.to("m/s").magnitude == pytest.approx(1.5)
    assert v2_part.to("m/s").magnitude == pytest.approx(4.5)


def test_momentum_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        linear_momentum("2 m", "5 m/s")
    with pytest.raises(pint.DimensionalityError):
        average_force("10 N", "1 s")
    with pytest.raises(pint.DimensionalityError):
        elastic_collision_velocities("1 kg", "2 m", "1 kg", "0 m/s")
    with pytest.raises(pint.DimensionalityError):
        collision_velocities("1 kg", "2 m/s", "1 kg", "0 m/s", restitution="0.5 m")

    with pytest.raises(PhysicalDomainError):
        linear_momentum("-1 kg", "5 m/s")
    with pytest.raises(PhysicalDomainError):
        impulse("10 N", "-1 s")
    with pytest.raises(PhysicalDomainError):
        average_force("10 N * s", "0 s")
    with pytest.raises(PhysicalDomainError):
        elastic_collision_velocities("0 kg", "5 m/s", "1 kg", "0 m/s")
    with pytest.raises(PhysicalDomainError):
        inelastic_collision_velocity("1 kg", "5 m/s", "-2 kg", "0 m/s")
    with pytest.raises(PhysicalDomainError):
        collision_velocities("1 kg", "5 m/s", "1 kg", "0 m/s", restitution=1.2)
    with pytest.raises(PhysicalDomainError):
        collision_velocities("1 kg", "5 m/s", "1 kg", "0 m/s", restitution=-0.1)


def test_gravitation_and_circular_dynamics() -> None:
    f_g = gravitational_force("1e6 kg", "2e6 kg", "2 m")
    expected_f = GRAVITATIONAL_CONSTANT.magnitude * 1e6 * 2e6 / 4.0
    assert str(f_g.units) == "newton"
    assert f_g.magnitude == pytest.approx(expected_f)

    g_earth = gravitational_field_strength("5.9722e24 kg", "6371 km")
    assert g_earth.to("m/s^2").magnitude == pytest.approx(9.82, rel=1e-2)

    v_orb = orbital_speed("1e4 kg", "1 m", gravitational_constant="1 m^3 / (kg * s^2)")
    assert v_orb.to("m/s").magnitude == pytest.approx(100.0)

    fc_v = centripetal_force("2 kg", "4 m", tangential_speed="6 m/s")
    assert fc_v.to("N").magnitude == pytest.approx(18.0)

    fc_w = centripetal_force("2 kg", "2 m", angular_velocity="60 rpm")
    # omega = 2*pi rad/s => Fc = 2 * (2*pi)^2 * 2 = 16*pi^2 N
    assert fc_w.to("N").magnitude == pytest.approx(157.91367041742973, rel=1e-9)


def test_gravitation_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        gravitational_force("10 kg", "10 kg", "2 s")
    with pytest.raises(pint.DimensionalityError):
        orbital_speed("1e24 kg", "10 m", gravitational_constant="10 N")
    with pytest.raises(pint.DimensionalityError):
        centripetal_force("2 kg", "4 m", angular_velocity="3 m/s")

    with pytest.raises(ValueError):
        centripetal_force("2 kg", "4 m")
    with pytest.raises(ValueError):
        centripetal_force("2 kg", "4 m", tangential_speed="3 m/s", angular_velocity="1 rad/s")

    with pytest.raises(PhysicalDomainError):
        gravitational_force("-1 kg", "2 kg", "1 m")
    with pytest.raises(PhysicalDomainError):
        gravitational_field_strength("10 kg", "0 m")
    with pytest.raises(PhysicalDomainError):
        orbital_speed("10 kg", "-5 m")
    with pytest.raises(PhysicalDomainError):
        orbital_speed("10 kg", "5 m", gravitational_constant="0 m^3 / (kg * s^2)")
    with pytest.raises(PhysicalDomainError):
        centripetal_force("-2 kg", "4 m", tangential_speed="6 m/s")
    with pytest.raises(PhysicalDomainError):
        centripetal_force("2 kg", "0 m", tangential_speed="6 m/s")
