"""Unit tests for physu.energy submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import GRAVITATIONAL_CONSTANT
from physu.energy import (
    elastic_potential_energy,
    elastic_work,
    gravitational_potential_energy,
    gravitational_work,
    kinetic_energy,
    kinetic_energy_from_momentum,
    mechanical_work,
    momentum_from_kinetic_energy,
    net_work,
    speed_from_kinetic_energy,
    universal_gravitational_potential_energy,
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
