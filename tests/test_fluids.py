"""Unit tests for physu.fluids submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError, fluids
from physu.constants import STANDARD_ATMOSPHERE, STANDARD_GRAVITY
from physu.fluids import (
    apparent_weight,
    bernoulli_pressure,
    buoyant_force,
    capillary_rise_height,
    continuity_speed,
    dynamic_pressure,
    floating_submerged_fraction,
    flow_speed,
    fluid_density,
    hydrostatic_gauge_pressure,
    hydrostatic_pressure,
    kinematic_viscosity,
    laplace_bubble_pressure,
    mass_flow_rate,
    pascal_hydraulic_force,
    poiseuille_flow_rate,
    pressure_head,
    reynolds_number,
    specific_gravity,
    stokes_drag_force,
    stokes_terminal_velocity,
    submerged_volume_floating,
    surface_tension_force,
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


def test_viscosity_reynolds_poiseuille_and_stokes() -> None:
    nu = kinematic_viscosity("1 cP", "1000 kg / m^3")
    assert str(nu.units) == "meter ** 2 / second"
    assert nu.to("cSt").magnitude == pytest.approx(1.0)

    re = reynolds_number("1000 kg / m^3", "0.5 m / s", "2 cm", "1 cP")
    assert re.dimensionless
    assert re.magnitude == pytest.approx(10000.0)

    q_pois = poiseuille_flow_rate("8 kPa", "2 mm", "1 cP", "1 m")
    assert str(q_pois.units) == "meter ** 3 / second"
    assert q_pois.magnitude == pytest.approx(math.pi * 8000.0 * (0.002**4) / (8.0 * 0.001 * 1.0))

    f_drag = stokes_drag_force("1.49 Pa * s", "1 mm", "10 cm / s")
    assert str(f_drag.units) == "newton"
    assert f_drag.magnitude == pytest.approx(6.0 * math.pi * 1.49 * 0.001 * 0.1)

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    v_term = stokes_terminal_velocity("0.5 mm", "2500 kg / m^3", "1260 kg / m^3", "1.49 Pa * s")
    assert str(v_term.units) == "meter / second"
    assert v_term.magnitude == pytest.approx(
        2.0 * (0.0005**2) * (2500.0 - 1260.0) * g_std / (9.0 * 1.49)
    )
    assert (
        stokes_terminal_velocity("0.5 mm", "1000 kg / m^3", "1000 kg / m^3", "1 cP").magnitude
        == 0.0
    )


def test_viscosity_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        kinematic_viscosity("1 Pa", "1000 kg / m^3")
    with pytest.raises(pint.DimensionalityError):
        reynolds_number("1000 kg", "1 m / s", "1 cm", "1 cP")
    with pytest.raises(pint.DimensionalityError):
        stokes_drag_force("1 cSt", "1 mm", "1 m / s")

    with pytest.raises(PhysicalDomainError):
        kinematic_viscosity("1 cP", "0 kg / m^3")
    with pytest.raises(PhysicalDomainError):
        reynolds_number("1000 kg / m^3", "1 m / s", "1 cm", "0 Pa * s")
    with pytest.raises(PhysicalDomainError):
        poiseuille_flow_rate("-100 Pa", "2 mm", "1 cP", "1 m")
    with pytest.raises(PhysicalDomainError):
        poiseuille_flow_rate("100 Pa", "2 mm", "1 cP", "0 m")
    with pytest.raises(PhysicalDomainError):
        stokes_drag_force("-1 cP", "1 mm", "1 m / s")
    with pytest.raises(PhysicalDomainError):
        stokes_terminal_velocity("1 mm", "2500 kg / m^3", "1000 kg / m^3", "0 Pa * s")


def test_capillarity_surface_tension_laplace_and_rise() -> None:
    f_surf = surface_tension_force("72.8 mN / m", "10 cm")
    assert str(f_surf.units) == "newton"
    assert f_surf.to("mN").magnitude == pytest.approx(7.28)

    # Soap bubble (2 interfaces): 4*gamma/R ; water droplet (1 interface): 2*gamma/R
    dp_bubble = laplace_bubble_pressure("25 dyne / cm", "1 cm", interfaces=2)
    assert str(dp_bubble.units) == "pascal"
    assert dp_bubble.magnitude == pytest.approx(4.0 * 0.025 / 0.01)

    dp_drop = laplace_bubble_pressure("72.8 mN / m", "1 mm", interfaces=1)
    assert dp_drop.magnitude == pytest.approx(2.0 * 0.0728 / 0.001)

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    h_water = capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0.5 mm")
    assert str(h_water.units) == "meter"
    assert h_water.magnitude == pytest.approx(2.0 * 0.0728 / (1000.0 * g_std * 0.0005))

    # Mercury non-wetting contact angle (140 deg) => negative capillary depression
    h_hg = capillary_rise_height(
        "486.5 mN / m",
        "13.5951 g / cm^3",
        "0.5 mm",
        contact_angle="140 deg",
    )
    assert h_hg.magnitude < 0.0
    assert capillary_rise_height(
        "72.8 mN / m",
        "1000 kg / m^3",
        "0.5 mm",
        contact_angle="90 deg",
    ).magnitude == 0.0


def test_capillarity_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        surface_tension_force("72 mN", "10 cm")
    with pytest.raises(pint.DimensionalityError):
        capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0.5 mm", contact_angle="30 m")

    with pytest.raises(PhysicalDomainError):
        surface_tension_force("-10 mN / m", "10 cm")
    with pytest.raises(PhysicalDomainError):
        laplace_bubble_pressure("25 mN / m", "0 cm")
    with pytest.raises(PhysicalDomainError):
        laplace_bubble_pressure("25 mN / m", "1 cm", interfaces=3)
    with pytest.raises(PhysicalDomainError):
        laplace_bubble_pressure("25 mN / m", "1 cm", interfaces=1.5)
    with pytest.raises(PhysicalDomainError):
        capillary_rise_height("72.8 mN / m", "0 kg / m^3", "0.5 mm")
    with pytest.raises(PhysicalDomainError):
        capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0 mm")
    with pytest.raises(PhysicalDomainError):
        capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0.5 mm", contact_angle="-5 deg")
    with pytest.raises(PhysicalDomainError):
        capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0.5 mm", contact_angle="185 deg")


def test_fluids_units_external_registry_and_exports() -> None:
    # Non-SI / CGS / imperial conversions: gal/min, psi, poise, stokes, g/mL
    v_gal = flow_speed("60 gal / min", "10 cm^2")
    assert v_gal.magnitude == pytest.approx(3.785411784)

    nu_stokes = kinematic_viscosity("1 poise", "1 g / mL")
    assert nu_stokes.to("stokes").magnitude == pytest.approx(1.0)

    # Rising air bubble in water (sphere_density < fluid_density) => negative velocity
    v_bubble = stokes_terminal_velocity("0.2 mm", "1.2 kg / m^3", "1000 kg / m^3", "1 cP")
    assert v_bubble.magnitude < 0.0

    # External Pint registry compatibility
    ext_ureg = pint.UnitRegistry()
    p_ext = hydrostatic_gauge_pressure(
        ext_ureg.Quantity(1000, "kg / m^3"),
        ext_ureg.Quantity(10, "m"),
    )
    assert p_ext.to("psi").magnitude == pytest.approx(
        hydrostatic_gauge_pressure("1000 kg / m^3", "10 m").to("psi").magnitude
    )

    # Non-finite rejection
    with pytest.raises(PhysicalDomainError):
        hydrostatic_gauge_pressure("nan kg / m^3", "10 m")
    with pytest.raises(PhysicalDomainError):
        reynolds_number("1000 kg / m^3", "inf m / s", "1 cm", "1 cP")

    # Package __all__ completeness across all 5 submodules
    for submodule in (
        fluids.hydrostatics,
        fluids.buoyancy,
        fluids.dynamics,
        fluids.viscosity,
        fluids.capillarity,
    ):
        for symbol in submodule.__all__:
            assert symbol in fluids.__all__
            assert hasattr(fluids, symbol)
