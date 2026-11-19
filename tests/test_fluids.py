"""Unit tests for physu.fluids submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import STANDARD_ATMOSPHERE, STANDARD_GRAVITY
from physu.fluids import (
    apparent_weight,
    bernoulli_pressure,
    buoyant_force,
    continuity_speed,
    dynamic_pressure,
    floating_submerged_fraction,
    flow_speed,
    fluid_density,
    hydrostatic_gauge_pressure,
    hydrostatic_pressure,
    mass_flow_rate,
    pascal_hydraulic_force,
    pressure_head,
    specific_gravity,
    submerged_volume_floating,
    torricelli_efflux_speed,
    volumetric_flow_rate,
)


def test_hydrostatics_calculations() -> None:
    rho = fluid_density("2.7 kg", "1 liter")
    assert str(rho.units) == "kilogram / meter ** 3"
    assert rho.to("g / cm^3").magnitude == pytest.approx(2.7)

    sg = specific_gravity("13.6 g / cm^3")
    assert sg.dimensionless
    assert sg.magnitude == pytest.approx(13.6)
    assert specific_gravity("0 kg / m^3").magnitude == 0.0

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    p_atm = float(STANDARD_ATMOSPHERE.to("Pa").magnitude)

    p_gauge = hydrostatic_gauge_pressure("1025 kg / m^3", "10 m")
    assert str(p_gauge.units) == "pascal"
    assert p_gauge.magnitude == pytest.approx(1025.0 * g_std * 10.0)

    p_abs = hydrostatic_pressure("1025 kg / m^3", "10 m")
    assert str(p_abs.units) == "pascal"
    assert p_abs.magnitude == pytest.approx(p_atm + 1025.0 * g_std * 10.0)

    f_out = pascal_hydraulic_force("250 N", "5 cm^2", "200 cm^2")
    assert str(f_out.units) == "newton"
    assert f_out.to("kN").magnitude == pytest.approx(10.0)

    # 1 atm of mercury (13.5951 g/cm^3) => ~760 mmHg head
    head_hg = pressure_head("1 atm", "13.5951 g / cm^3")
    assert str(head_hg.units) == "meter"
    assert head_hg.to("mm").magnitude == pytest.approx(760.0, rel=1e-3)


def test_hydrostatics_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        fluid_density("1 kg", "1 m^2")
    with pytest.raises(pint.DimensionalityError):
        specific_gravity("1000 kg")
    with pytest.raises(pint.DimensionalityError):
        hydrostatic_gauge_pressure("1000 kg / m^3", "10 s")
    with pytest.raises(pint.DimensionalityError):
        pascal_hydraulic_force("100 N", "5 cm", "20 cm^2")
    with pytest.raises(pint.DimensionalityError):
        pressure_head("100 N", "1000 kg / m^3")

    with pytest.raises(PhysicalDomainError):
        fluid_density("-1 kg", "1 liter")
    with pytest.raises(PhysicalDomainError):
        fluid_density("1 kg", "0 liter")
    with pytest.raises(PhysicalDomainError):
        specific_gravity("1000 kg / m^3", reference_density="0 kg / m^3")
    with pytest.raises(PhysicalDomainError):
        hydrostatic_gauge_pressure("1000 kg / m^3", "-5 m")
    with pytest.raises(PhysicalDomainError):
        hydrostatic_pressure("1000 kg / m^3", "5 m", surface_pressure="-1 Pa")
    with pytest.raises(PhysicalDomainError):
        pascal_hydraulic_force("100 N", "0 cm^2", "20 cm^2")
    with pytest.raises(PhysicalDomainError):
        pressure_head("1 bar", "0 kg / m^3")


def test_buoyancy_archimedes_and_floating() -> None:
    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)

    f_b = buoyant_force("1000 kg / m^3", "500 liter")
    assert str(f_b.units) == "newton"
    assert f_b.magnitude == pytest.approx(1000.0 * 0.5 * g_std)

    # 2 kg aluminum block (0.74 L) in water: W_app = (2 - 0.74) * g
    w_app = apparent_weight("2 kg", "0.74 liter", "1000 kg / m^3")
    assert str(w_app.units) == "newton"
    assert w_app.magnitude == pytest.approx((2.0 - 0.74) * g_std)
    assert apparent_weight("1 kg", "1 liter", "1000 kg / m^3").magnitude == pytest.approx(
        0.0, abs=1e-12
    )
    assert apparent_weight("1 kg", "0.001 m^3", "1000 kg / m^3").magnitude == 0.0

    # Iceberg (917 kg/m^3) in seawater (1025 kg/m^3) => ~89.46% submerged
    frac = floating_submerged_fraction("917 kg / m^3", "1025 kg / m^3")
    assert frac.dimensionless
    assert frac.magnitude == pytest.approx(917.0 / 1025.0)
    assert floating_submerged_fraction("0 kg / m^3", "1000 kg / m^3").magnitude == 0.0

    v_sub = submerged_volume_floating("4500 kg", "1000 kg / m^3")
    assert str(v_sub.units) == "meter ** 3"
    assert v_sub.magnitude == pytest.approx(4.5)


def test_buoyancy_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        buoyant_force("1000 kg / m^3", "2 m^2")
    with pytest.raises(pint.DimensionalityError):
        apparent_weight("2 kg", "1 liter", "1000 kg")
    with pytest.raises(pint.DimensionalityError):
        floating_submerged_fraction("900 kg", "1000 kg / m^3")

    with pytest.raises(PhysicalDomainError):
        buoyant_force("1000 kg / m^3", "-1 liter")
    with pytest.raises(PhysicalDomainError):
        apparent_weight("-2 kg", "1 liter", "1000 kg / m^3")
    with pytest.raises(PhysicalDomainError):
        floating_submerged_fraction("1100 kg / m^3", "1000 kg / m^3")
    with pytest.raises(PhysicalDomainError):
        floating_submerged_fraction("900 kg / m^3", "0 kg / m^3")
    with pytest.raises(PhysicalDomainError):
        submerged_volume_floating("10 kg", "0 kg / m^3")


def test_fluid_dynamics_continuity_bernoulli_and_torricelli() -> None:
    q = volumetric_flow_rate("20 cm^2", "2.5 m / s")
    assert str(q.units) == "meter ** 3 / second"
    assert q.to("L / min").magnitude == pytest.approx(300.0)

    v_from_q = flow_speed("300 L / min", "20 cm^2")
    assert str(v_from_q.units) == "meter / second"
    assert v_from_q.magnitude == pytest.approx(2.5)

    m_dot = mass_flow_rate("1000 kg / m^3", q)
    assert str(m_dot.units) == "kilogram / second"
    assert m_dot.magnitude == pytest.approx(5.0)

    v2 = continuity_speed("20 cm^2", "2.5 m / s", "5 cm^2")
    assert v2.magnitude == pytest.approx(10.0)

    q_dyn = dynamic_pressure("1.225 kg / m^3", "40 m / s")
    assert str(q_dyn.units) == "pascal"
    assert q_dyn.magnitude == pytest.approx(0.5 * 1.225 * 1600.0)

    # Venturi meter: P1 = 200 kPa, v1 = 2 m/s, v2 = 6 m/s, h1 = h2 => P2 = 200k - 0.5*1000*(36-4)
    p2 = bernoulli_pressure("200 kPa", "1000 kg / m^3", "2 m / s", "0 m", "6 m / s", "0 m")
    assert p2.to("kPa").magnitude == pytest.approx(184.0)

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    v_eff = torricelli_efflux_speed("5 m")
    assert str(v_eff.units) == "meter / second"
    assert v_eff.magnitude == pytest.approx(math.sqrt(2.0 * g_std * 5.0))


def test_fluid_dynamics_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        volumetric_flow_rate("2 m", "3 m / s")
    with pytest.raises(pint.DimensionalityError):
        flow_speed("10 m / s", "2 m^2")
    with pytest.raises(pint.DimensionalityError):
        bernoulli_pressure("100 N", "1000 kg / m^3", "1 m/s", "0 m", "2 m/s", "0 m")

    with pytest.raises(PhysicalDomainError):
        volumetric_flow_rate("-2 cm^2", "1 m / s")
    with pytest.raises(PhysicalDomainError):
        flow_speed("10 L / min", "0 cm^2")
    with pytest.raises(PhysicalDomainError):
        continuity_speed("10 cm^2", "2 m / s", "0 cm^2")
    with pytest.raises(PhysicalDomainError):
        dynamic_pressure("1000 kg / m^3", "-5 m / s")
    with pytest.raises(PhysicalDomainError):
        bernoulli_pressure("200 kPa", "1000 kg / m^3", "-2 m / s", "0 m", "6 m / s", "0 m")
    with pytest.raises(PhysicalDomainError):
        torricelli_efflux_speed("-2 m")
