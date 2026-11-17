"""Unit tests for physu.thermodynamics submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.thermodynamics import (
    linear_thermal_expansion,
    temperature_difference,
    to_celsius,
    to_fahrenheit,
    to_kelvin,
    volumetric_thermal_expansion,
)


def test_temperature_conversions_and_difference() -> None:
    t_k = to_kelvin("25 degC")
    assert str(t_k.units) == "kelvin"
    assert t_k.magnitude == pytest.approx(298.15)

    t_c = to_celsius("77 degF")
    assert str(t_c.units) == "degree_Celsius"
    assert t_c.magnitude == pytest.approx(25.0)

    t_f = to_fahrenheit("373.15 K")
    assert str(t_f.units) == "degree_Fahrenheit"
    assert t_f.magnitude == pytest.approx(212.0)

    assert to_kelvin("-273.15 degC").magnitude == pytest.approx(0.0)
    assert to_kelvin("-459.67 degF").magnitude == pytest.approx(0.0)

    dt_pos = temperature_difference("20 degC", "95 degC")
    assert str(dt_pos.units) == "kelvin"
    assert dt_pos.magnitude == pytest.approx(75.0)

    dt_neg = temperature_difference("212 degF", "32 degF")
    assert dt_neg.magnitude == pytest.approx(-100.0)


def test_thermal_expansion_calculations() -> None:
    # 50 m steel bridge span heated by 30 delta_degC with alpha = 1.2e-5 / K => 1.8 cm
    dl = linear_thermal_expansion("50 m", "1.2e-5 / delta_degC", "30 delta_degC")
    assert str(dl.units) == "meter"
    assert dl.to("cm").magnitude == pytest.approx(1.8)

    # 10 L fuel cooled by -20 K with beta = 9.5e-4 / K => -0.19 L
    dv = volumetric_thermal_expansion("10 liter", "9.5e-4 / K", "-20 K")
    assert dv.to("liter").magnitude == pytest.approx(-0.19)


def test_temperature_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        to_kelvin("25 m")
    with pytest.raises(pint.DimensionalityError):
        to_kelvin(25)
    with pytest.raises(pint.DimensionalityError):
        linear_thermal_expansion("10 m", "1.2e-5 / m", "20 K")

    with pytest.raises(PhysicalDomainError):
        to_kelvin("-1 K")
    with pytest.raises(PhysicalDomainError):
        to_kelvin("-300 degC")
    with pytest.raises(PhysicalDomainError):
        to_celsius("-500 degF")
    with pytest.raises(PhysicalDomainError):
        to_kelvin("10 delta_degC")
    with pytest.raises(PhysicalDomainError):
        temperature_difference("-280 degC", "20 degC")
    with pytest.raises(PhysicalDomainError):
        linear_thermal_expansion("-5 m", "1.2e-5 / K", "20 K")
    with pytest.raises(PhysicalDomainError):
        linear_thermal_expansion("5 m", "1.2e-5 / K", "20 degC")
    with pytest.raises(PhysicalDomainError):
        volumetric_thermal_expansion("-1 liter", "9e-4 / K", "10 K")
