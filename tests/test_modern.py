"""Unit tests for physu.modern submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import ELECTRON_MASS, SPEED_OF_LIGHT
from physu.modern import (
    energy_momentum_relation,
    length_contraction,
    lorentz_factor,
    relativistic_kinetic_energy,
    relativistic_momentum,
    relativistic_total_energy,
    relativistic_velocity_addition,
    rest_energy,
    time_dilation,
)


def test_relativity_lorentz_time_length_and_velocity_addition() -> None:
    # At v = 0.6 c => gamma = 1.25
    gamma = lorentz_factor("0.6 c")
    assert gamma.dimensionless
    assert gamma.magnitude == pytest.approx(1.25)
    assert lorentz_factor("0 m / s").magnitude == pytest.approx(1.0)

    dt = time_dilation("2.2 us", "0.6 c")
    assert str(dt.units) == "second"
    assert dt.to("us").magnitude == pytest.approx(2.75)

    length = length_contraction("100 m", "0.6 c")
    assert str(length.units) == "meter"
    assert length.magnitude == pytest.approx(80.0)

    # 0.6 c + 0.6 c => 1.2 / 1.36 c = 15/17 c
    u = relativistic_velocity_addition("0.6 c", "0.6 c")
    c_val = float(SPEED_OF_LIGHT.to("m / s").magnitude)
    assert str(u.units) == "meter / second"
    assert u.magnitude == pytest.approx((15.0 / 17.0) * c_val)
    assert relativistic_velocity_addition("0.6 c", "-0.6 c").magnitude == 0.0


def test_relativity_energy_and_momentum() -> None:
    e0 = rest_energy(ELECTRON_MASS)
    assert str(e0.units) == "joule"
    assert e0.to("MeV").magnitude == pytest.approx(0.51099895, rel=1e-6)

    e_tot = relativistic_total_energy(ELECTRON_MASS, "0.8 c")  # gamma = 5/3
    k_rel = relativistic_kinetic_energy(ELECTRON_MASS, "0.8 c")
    p_rel = relativistic_momentum(ELECTRON_MASS, "0.8 c")

    assert e_tot.to("MeV").magnitude == pytest.approx((5.0 / 3.0) * e0.to("MeV").magnitude)
    assert k_rel.to("MeV").magnitude == pytest.approx((2.0 / 3.0) * e0.to("MeV").magnitude)
    assert str(p_rel.units) == "kilogram * meter / second"

    # E^2 = (p c)^2 + (m c^2)^2 matches gamma * m * c^2
    e_disp = energy_momentum_relation(ELECTRON_MASS, p_rel)
    assert e_disp.magnitude == pytest.approx(e_tot.magnitude)

    # Massless photon dispersion E = p * c
    e_photon = energy_momentum_relation("0 kg", "1e-27 kg * m / s")
    assert e_photon.magnitude == pytest.approx(1e-27 * float(SPEED_OF_LIGHT.to("m / s").magnitude))
    assert relativistic_kinetic_energy(ELECTRON_MASS, "0 m / s").magnitude == 0.0


def test_relativity_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        lorentz_factor("0.5 m")
    with pytest.raises(pint.DimensionalityError):
        time_dilation("1 m", "0.5 c")
    with pytest.raises(pint.DimensionalityError):
        rest_energy("1 J")
    with pytest.raises(pint.DimensionalityError):
        energy_momentum_relation(ELECTRON_MASS, "1 N")

    with pytest.raises(PhysicalDomainError):
        lorentz_factor("1.0 c")
    with pytest.raises(PhysicalDomainError):
        lorentz_factor("-1.2 c")
    with pytest.raises(PhysicalDomainError):
        time_dilation("-1 s", "0.5 c")
    with pytest.raises(PhysicalDomainError):
        length_contraction("-10 m", "0.5 c")
    with pytest.raises(PhysicalDomainError):
        relativistic_velocity_addition("1.0 c", "0.2 c")
    with pytest.raises(PhysicalDomainError):
        rest_energy("-1 kg")
    with pytest.raises(PhysicalDomainError):
        relativistic_momentum(ELECTRON_MASS, "1.0 c")
    with pytest.raises(PhysicalDomainError):
        lorentz_factor("0.5 m / s", speed_of_light="0 m / s")
    assert math.isclose(lorentz_factor("-0.6 c").magnitude, 1.25)
