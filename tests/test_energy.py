"""Unit tests for physu.energy submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.energy import (
    elastic_work,
    gravitational_work,
    mechanical_work,
    net_work,
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
