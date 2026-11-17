"""Unit tests for physu.thermodynamics submodules."""

from __future__ import annotations

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import BOLTZMANN_CONSTANT, MOLAR_GAS_CONSTANT
from physu.thermodynamics import (
    average_molecular_kinetic_energy,
    combined_gas_law,
    equilibrium_temperature,
    ideal_gas_moles,
    ideal_gas_pressure,
    ideal_gas_temperature,
    ideal_gas_volume,
    latent_heat,
    linear_thermal_expansion,
    rms_molecular_speed,
    sensible_heat,
    specific_heat_capacity,
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


def test_calorimetry_calculations_and_conversions() -> None:
    q_sens = sensible_heat("500 g", "4184 J / (kg * K)", "20 delta_degC")
    assert str(q_sens.units) == "joule"
    assert q_sens.to("kJ").magnitude == pytest.approx(41.84)

    q_cool = sensible_heat("1 kg", "1 cal / (g * delta_degC)", "-10 K")
    assert q_cool.to("kcal").magnitude == pytest.approx(-10.0)

    q_lat = latent_heat("2 kg", "334 kJ / kg")
    assert q_lat.to("kJ").magnitude == pytest.approx(668.0)

    c_spec = specific_heat_capacity("8368 J", "200 g", "10 delta_degC")
    assert c_spec.to("J / (kg * K)").magnitude == pytest.approx(4184.0)

    t_eq = equilibrium_temperature(
        [
            ("200 g", "4184 J / (kg * K)", "80 degC"),
            ("300 g", "4184 J / (kg * K)", "20 degC"),
        ]
    )
    assert t_eq.to("degC").magnitude == pytest.approx(44.0)


def test_calorimetry_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        sensible_heat("1 m", "4184 J / (kg * K)", "10 K")
    with pytest.raises(pint.DimensionalityError):
        latent_heat("2 kg", "334 J")
    with pytest.raises(pint.DimensionalityError):
        specific_heat_capacity("100 N", "1 kg", "10 K")

    with pytest.raises(PhysicalDomainError):
        sensible_heat("-1 kg", "4184 J / (kg * K)", "10 K")
    with pytest.raises(PhysicalDomainError):
        sensible_heat("1 kg", "4184 J / (kg * K)", "10 degC")
    with pytest.raises(PhysicalDomainError):
        latent_heat("1 kg", "-100 J / kg")
    with pytest.raises(PhysicalDomainError):
        specific_heat_capacity("100 J", "0 kg", "10 K")
    with pytest.raises(PhysicalDomainError):
        specific_heat_capacity("100 J", "1 kg", "0 K")
    with pytest.raises(PhysicalDomainError):
        specific_heat_capacity("100 J", "1 kg", "-10 K")
    with pytest.raises(PhysicalDomainError):
        equilibrium_temperature([])
    with pytest.raises(PhysicalDomainError):
        equilibrium_temperature([("0 kg", "4184 J / (kg * K)", "20 degC")])
    with pytest.raises(PhysicalDomainError):
        equilibrium_temperature([("1 kg", "4184 J / (kg * K)", "-300 degC")])


def test_ideal_gas_and_kinetic_theory() -> None:
    # 1 mol at 0 degC and 1 atm => ~22.414 L
    v_stp = ideal_gas_volume("1 mol", "0 degC", "1 atm")
    assert v_stp.to("liter").magnitude == pytest.approx(22.41396954, rel=1e-6)

    p = ideal_gas_pressure("2 mol", "300 K", "10 liter")
    expected_p = 2.0 * MOLAR_GAS_CONSTANT.magnitude * 300.0 / 0.01
    assert p.to("Pa").magnitude == pytest.approx(expected_p)

    n = ideal_gas_moles("101.325 kPa", "22.41396954 liter", "273.15 K")
    assert n.to("mol").magnitude == pytest.approx(1.0, rel=1e-6)

    t = ideal_gas_temperature("1 atm", "22.41396954 liter", "1 mol")
    assert t.to("degC").magnitude == pytest.approx(0.0, abs=1e-4)

    # Combined gas law: double temp and halve volume => 4x pressure
    p2 = combined_gas_law(
        "1 atm",
        "2 liter",
        "300 K",
        final_volume="1 liter",
        final_temperature="600 K",
    )
    assert p2.to("atm").magnitude == pytest.approx(4.0)

    v2 = combined_gas_law(
        "2 bar",
        "3 liter",
        "300 K",
        final_pressure="1 bar",
        final_temperature="300 K",
    )
    assert v2.to("liter").magnitude == pytest.approx(6.0)

    t2 = combined_gas_law(
        "1 atm",
        "4 liter",
        "27 degC",
        final_pressure="2 atm",
        final_volume="2 liter",
    )
    assert t2.to("K").magnitude == pytest.approx(300.15)

    # N2 (28.0134 g/mol) at 300 K => ~516.9 m/s
    v_rms = rms_molecular_speed("300 K", "28.0134 g / mol")
    assert v_rms.to("m/s").magnitude == pytest.approx(516.87, rel=1e-3)

    k_avg = average_molecular_kinetic_energy("300 K")
    assert k_avg.to("J").magnitude == pytest.approx(1.5 * BOLTZMANN_CONSTANT.magnitude * 300.0)


def test_ideal_gas_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        ideal_gas_pressure("1 kg", "300 K", "1 liter")
    with pytest.raises(pint.DimensionalityError):
        rms_molecular_speed("300 K", "28 g")

    with pytest.raises(ValueError):
        combined_gas_law("1 atm", "1 liter", "300 K", final_pressure="2 atm")
    with pytest.raises(ValueError):
        combined_gas_law(
            "1 atm",
            "1 liter",
            "300 K",
            final_pressure="2 atm",
            final_volume="1 liter",
            final_temperature="300 K",
        )

    with pytest.raises(PhysicalDomainError):
        ideal_gas_pressure("1 mol", "300 K", "0 liter")
    with pytest.raises(PhysicalDomainError):
        ideal_gas_volume("1 mol", "300 K", "0 atm")
    with pytest.raises(PhysicalDomainError):
        ideal_gas_moles("1 atm", "1 liter", "0 K")
    with pytest.raises(PhysicalDomainError):
        ideal_gas_temperature("1 atm", "1 liter", "0 mol")
    with pytest.raises(PhysicalDomainError):
        rms_molecular_speed("-5 K", "28 g / mol")
    with pytest.raises(PhysicalDomainError):
        rms_molecular_speed("300 K", "0 g / mol")
    with pytest.raises(PhysicalDomainError):
        average_molecular_kinetic_energy("-300 degC")
