"""Unit tests for physu.dynamics submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError, ureg
from physu.dynamics import (
    acceleration,
    critical_incline_angle,
    force,
    friction_coefficient,
    inclined_normal_force,
    inclined_parallel_force,
    kinetic_friction,
    mass,
    max_static_friction,
    normal_force,
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
