"""Unit tests for physu.dynamics submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError, ureg
from physu.dynamics import (
    acceleration,
    force,
    inclined_normal_force,
    inclined_parallel_force,
    mass,
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
