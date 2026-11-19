"""Unit tests for physu.fluids submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import STANDARD_ATMOSPHERE, STANDARD_GRAVITY
from physu.fluids import (
    fluid_density,
    hydrostatic_gauge_pressure,
    hydrostatic_pressure,
    pascal_hydraulic_force,
    pressure_head,
    specific_gravity,
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
