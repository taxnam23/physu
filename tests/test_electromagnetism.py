"""Unit tests for physu.electromagnetism submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import COULOMB_CONSTANT, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from physu.electromagnetism import (
    coulomb_force,
    electric_field_point_charge,
    electric_flux,
    electric_force,
    electric_potential_energy,
    electric_potential_point_charge,
    electric_work,
    gauss_electric_flux,
    uniform_electric_field,
)


def test_coulomb_and_electric_field_calculations() -> None:
    k_e = float(COULOMB_CONSTANT.to("N * m^2 / C^2").magnitude)

    # Like charges: +2 uC and +3 uC at 10 cm in vacuum -> repulsive force
    f_rep = coulomb_force("2 uC", "3 uC", "10 cm")
    assert str(f_rep.units) == "newton"
    assert f_rep.magnitude == pytest.approx(k_e * 2e-6 * 3e-6 / 0.1**2)

    # Opposite charges in dielectric (eps_r = 4) -> attractive and scaled down
    f_attr = coulomb_force("2 uC", "-3 uC", "10 cm", relative_permittivity=4.0)
    assert f_attr.magnitude == pytest.approx(-f_rep.magnitude / 4.0)

    # Zero charge yields clean 0.0 N
    f_zero = coulomb_force("0 C", "-3 uC", "10 cm")
    assert f_zero.magnitude == 0.0

    # Point charge electric field (V / m == N / C) and electric force F = q * E
    e_field = electric_field_point_charge("-5 nC", "20 cm")
    assert str(e_field.units) == "volt / meter"
    assert e_field.to("N / C").magnitude == pytest.approx(-k_e * 5e-9 / 0.2**2)

    f_on_test = electric_force("4 uC", "2.5 kN / C")
    assert f_on_test.to("mN").magnitude == pytest.approx(10.0)


def test_uniform_field_potential_energy_and_work() -> None:
    k_e = float(COULOMB_CONSTANT.to("N * m^2 / C^2").magnitude)

    e_plates = uniform_electric_field("1.5 kV", "5 mm")
    assert str(e_plates.units) == "volt / meter"
    assert e_plates.to("kV / m").magnitude == pytest.approx(300.0)

    v_point = electric_potential_point_charge("4 uC", "2 m")
    assert str(v_point.units) == "volt"
    assert v_point.magnitude == pytest.approx(k_e * 4e-6 / 2.0)

    u_pair = electric_potential_energy("2 uC", "-3 uC", "0.5 m", relative_permittivity=2.0)
    assert str(u_pair.units) == "joule"
    assert u_pair.magnitude == pytest.approx(k_e * 2e-6 * (-3e-6) / (2.0 * 0.5))

    # Electron accelerated across +1 kV potential increase:
    # W_e = -(-e)(+1000 V) = +1 keV
    w_elec = electric_work(-ELEMENTARY_CHARGE, "1 kV")
    assert str(w_elec.units) == "joule"
    assert w_elec.to("keV").magnitude == pytest.approx(1.0)
    assert electric_work("0 C", "100 V").magnitude == 0.0


def test_electric_flux_and_gauss_law() -> None:
    flux_perp = electric_flux("500 N / C", "200 cm^2")
    assert str(flux_perp.units) == "meter * volt"
    assert flux_perp.magnitude == pytest.approx(10.0)

    flux_angled = electric_flux("500 V / m", "2 m^2", angle="60 deg")
    assert flux_angled.magnitude == pytest.approx(500.0)

    flux_zero = electric_flux("500 V / m", "2 m^2", angle="90 deg")
    assert flux_zero.magnitude == 0.0

    q_enc = 1.77083756376e-9
    gauss_flux = gauss_electric_flux(f"{q_enc} C")
    eps_0 = float(VACUUM_PERMITTIVITY.to("F / m").magnitude)
    assert str(gauss_flux.units) == "meter * volt"
    assert gauss_flux.magnitude == pytest.approx(q_enc / eps_0)

    # Sphere Gauss check: E(r) * 4*pi*r^2 == Q_enc / (eps_0 * eps_r)
    q = "3 uC"
    r = 0.4
    eps_r = 2.5
    e_r = electric_field_point_charge(q, f"{r} m", relative_permittivity=eps_r)
    sphere_flux = electric_flux(e_r, f"{4.0 * math.pi * r**2} m^2")
    assert sphere_flux.magnitude == pytest.approx(
        gauss_electric_flux(q, relative_permittivity=eps_r).magnitude
    )


def test_electrostatics_validation_errors() -> None:
    # Dimensionality mismatches
    with pytest.raises(pint.DimensionalityError):
        coulomb_force("2 A", "3 C", "1 m")
    with pytest.raises(pint.DimensionalityError):
        electric_field_point_charge("2 C", "1 s")
    with pytest.raises(pint.DimensionalityError):
        electric_force("1 C", "10 T")
    with pytest.raises(pint.DimensionalityError):
        uniform_electric_field("12 A", "1 cm")
    with pytest.raises(pint.DimensionalityError):
        electric_potential_point_charge("1 C", "2 m", relative_permittivity="2 m")
    with pytest.raises(pint.DimensionalityError):
        electric_work("1 C", "10 N")
    with pytest.raises(pint.DimensionalityError):
        electric_flux("100 V / m", "2 m")
    with pytest.raises(pint.DimensionalityError):
        electric_flux("100 V / m", "2 m^2", angle="10 m")
    with pytest.raises(pint.DimensionalityError):
        gauss_electric_flux("5 V")

    # Physical domain violations
    with pytest.raises(PhysicalDomainError):
        coulomb_force("1 C", "1 C", "0 m")
    with pytest.raises(PhysicalDomainError):
        coulomb_force("1 C", "1 C", "-0.5 m")
    with pytest.raises(PhysicalDomainError):
        electric_field_point_charge("1 C", "1 m", relative_permittivity=0)
    with pytest.raises(PhysicalDomainError):
        electric_field_point_charge("1 C", "1 m", relative_permittivity=-2)
    with pytest.raises(PhysicalDomainError):
        uniform_electric_field("12 V", "0 m")
    with pytest.raises(PhysicalDomainError):
        electric_potential_point_charge("1 C", "-1 cm")
    with pytest.raises(PhysicalDomainError):
        electric_flux("100 V / m", "-2 m^2")
    with pytest.raises(PhysicalDomainError):
        gauss_electric_flux("1 uC", relative_permittivity=0)
