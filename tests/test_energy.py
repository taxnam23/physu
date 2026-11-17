"""Unit tests for physu.energy submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import GRAVITATIONAL_CONSTANT
from physu.energy import (
    average_power,
    elastic_potential_energy,
    elastic_work,
    escape_speed,
    final_height_conservation,
    final_speed_conservation,
    force_from_power,
    gravitational_potential_energy,
    gravitational_work,
    instantaneous_power,
    kinetic_energy,
    kinetic_energy_from_momentum,
    mechanical_efficiency,
    mechanical_energy,
    mechanical_work,
    momentum_from_kinetic_energy,
    net_work,
    non_conservative_work,
    speed_from_kinetic_energy,
    universal_gravitational_potential_energy,
    velocity_from_power,
    work_from_power,
)


def test_work_calculations_and_unit_conversions() -> None:
    w_parallel = mechanical_work("50 N", "4 m")
    assert str(w_parallel.units) == "joule"
    assert w_parallel.magnitude == pytest.approx(200.0)

    w_angled = mechanical_work("20 N", "5 m", angle="60 deg")
    assert w_angled.to("kJ").magnitude == pytest.approx(0.05)

    w_perp = mechanical_work("100 N", "10 m", angle="90 deg")
    assert w_perp.magnitude == 0.0
    assert str(w_perp.magnitude) == "0.0"

    w_opp = mechanical_work("10 N", "3 m", angle="180 deg")
    assert w_opp.to("J").magnitude == pytest.approx(-30.0)

    w_net = net_work("1 kJ", "2.5 kJ")
    assert w_net.to("J").magnitude == pytest.approx(1500.0)

    w_net_slow = net_work("500 J", "200 J")
    assert w_net_slow.to("J").magnitude == pytest.approx(-300.0)

    w_grav_up = gravitational_work("2 kg", "5 m", gravity="10 m/s^2")
    assert w_grav_up.to("J").magnitude == pytest.approx(-100.0)

    w_grav_down = gravitational_work("2 kg", "-5 m", gravity="10 m/s^2")
    assert w_grav_down.to("J").magnitude == pytest.approx(100.0)

    assert gravitational_work("2 kg", "0 m").magnitude == 0.0

    # Elastic work returning from 20 cm to 0 cm with k = 200 N/m => 0.5*200*(0.04 - 0) = 4 J
    w_spring = elastic_work("200 N/m", "20 cm", "0 cm")
    assert w_spring.to("J").magnitude == pytest.approx(4.0)


def test_work_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        mechanical_work("10 kg", "5 m")
    with pytest.raises(pint.DimensionalityError):
        mechanical_work("10 N", "5 s")
    with pytest.raises(pint.DimensionalityError):
        mechanical_work("10 N", "5 m", angle="2 m")
    with pytest.raises(pint.DimensionalityError):
        net_work("10 N", "20 J")
    with pytest.raises(pint.DimensionalityError):
        elastic_work("100 N", "0.1 m", "0 m")

    with pytest.raises(PhysicalDomainError):
        net_work("-10 J", "20 J")
    with pytest.raises(PhysicalDomainError):
        net_work("10 J", "-5 J")
    with pytest.raises(PhysicalDomainError):
        gravitational_work("-2 kg", "5 m")
    with pytest.raises(PhysicalDomainError):
        gravitational_work("2 kg", "5 m", gravity="0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        elastic_work("-100 N/m", "0.1 m", "0 m")


def test_kinetic_energy_calculations_and_conversions() -> None:
    k = kinetic_energy("2000 g", "-10 m/s")
    assert str(k.units) == "joule"
    assert k.magnitude == pytest.approx(100.0)

    k_from_p = kinetic_energy_from_momentum("20 N * s", "2 kg")
    assert k_from_p.to("J").magnitude == pytest.approx(100.0)

    v = speed_from_kinetic_energy("1 kJ", "5 kg")
    assert v.to("m/s").magnitude == pytest.approx(20.0)

    p = momentum_from_kinetic_energy("100 J", "2 kg")
    assert p.to("kg * m / s").magnitude == pytest.approx(20.0)
    assert p.to("N * s").magnitude == pytest.approx(20.0)

    k_zero = momentum_from_kinetic_energy("100 J", "0 kg")
    assert k_zero.magnitude == 0.0


def test_kinetic_energy_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        kinetic_energy("2 m", "10 m/s")
    with pytest.raises(pint.DimensionalityError):
        kinetic_energy_from_momentum("20 J", "2 kg")
    with pytest.raises(pint.DimensionalityError):
        speed_from_kinetic_energy("100 N", "2 kg")

    with pytest.raises(PhysicalDomainError):
        kinetic_energy("-2 kg", "10 m/s")
    with pytest.raises(PhysicalDomainError):
        kinetic_energy_from_momentum("20 N * s", "0 kg")
    with pytest.raises(PhysicalDomainError):
        kinetic_energy_from_momentum("20 N * s", "-2 kg")
    with pytest.raises(PhysicalDomainError):
        speed_from_kinetic_energy("-100 J", "2 kg")
    with pytest.raises(PhysicalDomainError):
        speed_from_kinetic_energy("100 J", "0 kg")
    with pytest.raises(PhysicalDomainError):
        momentum_from_kinetic_energy("-50 J", "2 kg")
    with pytest.raises(PhysicalDomainError):
        momentum_from_kinetic_energy("50 J", "-2 kg")


def test_potential_energy_calculations_and_conversions() -> None:
    u_g = gravitational_potential_energy("2 kg", "5 m", gravity="10 m/s^2")
    assert str(u_g.units) == "joule"
    assert u_g.magnitude == pytest.approx(100.0)

    u_g_below = gravitational_potential_energy("2 kg", "-3 m", gravity="10 m/s^2")
    assert u_g_below.to("J").magnitude == pytest.approx(-60.0)

    u_univ = universal_gravitational_potential_energy("1e6 kg", "2e6 kg", "4 m")
    expected_u = -GRAVITATIONAL_CONSTANT.magnitude * 1e6 * 2e6 / 4.0
    assert u_univ.to("J").magnitude == pytest.approx(expected_u)

    u_zero = universal_gravitational_potential_energy("0 kg", "2 kg", "1 m")
    assert u_zero.magnitude == 0.0
    assert str(u_zero.magnitude) == "0.0"

    u_el = elastic_potential_energy("200 N/m", "-10 cm")
    assert u_el.to("J").magnitude == pytest.approx(1.0)


def test_potential_energy_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        gravitational_potential_energy("2 m", "5 m")
    with pytest.raises(pint.DimensionalityError):
        universal_gravitational_potential_energy("10 kg", "10 kg", "2 s")
    with pytest.raises(pint.DimensionalityError):
        elastic_potential_energy("100 N", "0.1 m")

    with pytest.raises(PhysicalDomainError):
        gravitational_potential_energy("-2 kg", "5 m")
    with pytest.raises(PhysicalDomainError):
        gravitational_potential_energy("2 kg", "5 m", gravity="0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        universal_gravitational_potential_energy("-1 kg", "2 kg", "1 m")
    with pytest.raises(PhysicalDomainError):
        universal_gravitational_potential_energy("1 kg", "2 kg", "0 m")
    with pytest.raises(PhysicalDomainError):
        elastic_potential_energy("-50 N/m", "0.1 m")


def test_conservation_calculations_and_conversions() -> None:
    e_total = mechanical_energy("250 J", "-50 J")
    assert str(e_total.units) == "joule"
    assert e_total.magnitude == pytest.approx(200.0)

    w_nc = non_conservative_work("1 kJ", "850 J")
    assert w_nc.to("J").magnitude == pytest.approx(-150.0)

    # Drop from 20 m with g = 10 m/s^2 starting from rest => v_f = 20 m/s
    v_f = final_speed_conservation("0 m/s", "20 m", "0 m", gravity="10 m/s^2")
    assert v_f.to("m/s").magnitude == pytest.approx(20.0)

    # Launch upward at 20 m/s from 5 m to final speed 10 m/s => h_f = 5 + (400 - 100)/20 = 20 m
    h_f = final_height_conservation("20 m/s", "5 m", "10 m/s", gravity="10 m/s^2")
    assert h_f.to("m").magnitude == pytest.approx(20.0)

    # Escape speed from Earth surface ~ 11.186 km/s
    v_esc = escape_speed("5.9722e24 kg", "6371 km")
    assert v_esc.to("km/s").magnitude == pytest.approx(11.186, rel=1e-3)

    eta = mechanical_efficiency("400 J", "0.5 kJ")
    assert eta.dimensionless
    assert eta.magnitude == pytest.approx(0.8)
    assert eta.to("%").magnitude == pytest.approx(80.0)


def test_conservation_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        mechanical_energy("10 N", "5 J")
    with pytest.raises(pint.DimensionalityError):
        final_speed_conservation("10 m", "5 m", "0 m")
    with pytest.raises(pint.DimensionalityError):
        mechanical_efficiency("10 W", "20 J")

    with pytest.raises(PhysicalDomainError):
        mechanical_energy("-10 J", "50 J")
    with pytest.raises(PhysicalDomainError):
        final_speed_conservation("10 m/s", "0 m", "10 m", gravity="10 m/s^2")
    with pytest.raises(PhysicalDomainError):
        final_speed_conservation("-5 m/s", "10 m", "0 m")
    with pytest.raises(PhysicalDomainError):
        final_height_conservation("10 m/s", "5 m", "-2 m/s")
    with pytest.raises(PhysicalDomainError):
        escape_speed("1e24 kg", "0 m")
    with pytest.raises(PhysicalDomainError):
        mechanical_efficiency("-10 J", "100 J")
    with pytest.raises(PhysicalDomainError):
        mechanical_efficiency("10 J", "0 J")
    with pytest.raises(PhysicalDomainError):
        mechanical_efficiency("120 J", "100 J")


def test_power_calculations_and_unit_conversions() -> None:
    p_avg = average_power("3 kJ", "1 min")
    assert str(p_avg.units) == "watt"
    assert p_avg.magnitude == pytest.approx(50.0)

    p_inst = instantaneous_power("100 N", "72 km/h")
    assert p_inst.to("kW").magnitude == pytest.approx(2.0)

    p_angled = instantaneous_power("100 N", "10 m/s", angle="60 deg")
    assert p_angled.to("W").magnitude == pytest.approx(500.0)

    p_perp = instantaneous_power("100 N", "10 m/s", angle="90 deg")
    assert p_perp.magnitude == 0.0
    assert str(p_perp.magnitude) == "0.0"

    w_kwh = work_from_power("2 kW", "3 hour")
    assert w_kwh.to("kWh").magnitude == pytest.approx(6.0)
    assert work_from_power("100 W", "0 s").magnitude == 0.0

    # 1 hp ~= 745.7 W
    f_hp = force_from_power("1 hp", "10 m/s")
    assert f_hp.to("N").magnitude == pytest.approx(74.56998715822702, rel=1e-6)

    v_pow = velocity_from_power("1 kW", "200 N", angle="60 deg")
    assert v_pow.to("m/s").magnitude == pytest.approx(10.0)


def test_power_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        average_power("10 N", "2 s")
    with pytest.raises(pint.DimensionalityError):
        instantaneous_power("10 kg", "5 m/s")
    with pytest.raises(pint.DimensionalityError):
        work_from_power("100 J", "5 s")
    with pytest.raises(pint.DimensionalityError):
        force_from_power("100 W", "5 s")

    with pytest.raises(PhysicalDomainError):
        average_power("100 J", "0 s")
    with pytest.raises(PhysicalDomainError):
        average_power("100 J", "-2 s")
    with pytest.raises(PhysicalDomainError):
        work_from_power("100 W", "-1 s")
    with pytest.raises(PhysicalDomainError):
        force_from_power("100 W", "0 m/s")
    with pytest.raises(PhysicalDomainError):
        force_from_power("100 W", "5 m/s", angle="90 deg")
    with pytest.raises(PhysicalDomainError):
        velocity_from_power("100 W", "0 N")
    with pytest.raises(PhysicalDomainError):
        velocity_from_power("100 W", "20 N", angle="90 deg")
