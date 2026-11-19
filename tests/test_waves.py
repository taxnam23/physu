"""Unit tests for physu.waves submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError, waves
from physu.constants import MOLAR_GAS_CONSTANT, STANDARD_GRAVITY
from physu.waves import (
    acoustic_impedance,
    acoustic_reflection_coefficient,
    beat_frequency,
    closed_pipe_harmonic_frequency,
    closed_pipe_harmonic_wavelength,
    doppler_frequency,
    fundamental_tension_for_frequency,
    intensity_from_sound_level,
    mach_angle,
    mach_number,
    open_pipe_harmonic_frequency,
    physical_pendulum_period,
    relativistic_doppler_frequency,
    shm_acceleration,
    shm_displacement,
    shm_energy,
    shm_max_acceleration,
    shm_max_speed,
    shm_speed_at_position,
    shm_velocity,
    simple_pendulum_period,
    sound_intensity_level,
    sound_pressure_level,
    speed_of_sound_ideal_gas,
    spherical_wave_intensity,
    spring_mass_angular_frequency,
    spring_mass_period,
    string_harmonic_frequency,
    string_harmonic_wavelength,
    string_wave_power,
    string_wave_speed,
    wave_frequency,
    wave_intensity,
    wave_phase_speed,
    wave_speed,
    wavelength,
    wavenumber,
)


def test_spring_mass_and_pendulums() -> None:
    omega = spring_mass_angular_frequency("400 N / m", "1 kg")
    assert str(omega.units) == "radian / second"
    assert omega.magnitude == pytest.approx(20.0)

    t_spring = spring_mass_period("500 g", "200 N / m")
    assert str(t_spring.units) == "second"
    assert t_spring.magnitude == pytest.approx(2.0 * math.pi * math.sqrt(0.5 / 200.0))

    g_std = float(STANDARD_GRAVITY.to("m / s^2").magnitude)
    t_pend = simple_pendulum_period("100 cm")
    assert str(t_pend.units) == "second"
    assert t_pend.magnitude == pytest.approx(2.0 * math.pi * math.sqrt(1.0 / g_std))

    # Uniform rod of mass 1.5 kg, length 1.2 m pivoted at one end: I = 1/3 m L^2, d = L/2
    inertia = (1.0 / 3.0) * 1.5 * (1.2**2)
    t_phys = physical_pendulum_period(
        f"{inertia} kg * m^2",
        "1.5 kg",
        "0.6 m",
        gravity="9.81 m/s^2",
    )
    assert str(t_phys.units) == "second"
    assert t_phys.magnitude == pytest.approx(2.0 * math.pi * math.sqrt((2.0 * 1.2) / (3.0 * 9.81)))


def test_shm_kinematics_and_energy() -> None:
    # At t = 0 with phase = 90 deg: x = 0, v = -A * omega, a = 0
    x_90 = shm_displacement("10 cm", "4 rad / s", "0 s", phase="90 deg")
    v_90 = shm_velocity("10 cm", "4 rad / s", "0 s", phase="90 deg")
    a_90 = shm_acceleration("10 cm", "4 rad / s", "0 s", phase="90 deg")
    assert str(x_90.units) == "meter"
    assert x_90.magnitude == 0.0
    assert str(v_90.units) == "meter / second"
    assert v_90.magnitude == pytest.approx(-0.4)
    assert str(a_90.units) == "meter / second ** 2"
    assert a_90.magnitude == 0.0

    # At t = 0 with phase = 0: x = A, v = 0, a = -A * omega^2
    assert shm_displacement("10 cm", "4 rad / s", "0 s").to("cm").magnitude == pytest.approx(10.0)
    assert shm_velocity("10 cm", "4 rad / s", "0 s").magnitude == 0.0
    assert shm_acceleration("10 cm", "4 rad / s", "0 s").magnitude == pytest.approx(-1.6)

    assert shm_max_speed("10 cm", "4 rad / s").magnitude == pytest.approx(0.4)
    assert shm_max_acceleration("10 cm", "4 rad / s").magnitude == pytest.approx(1.6)
    assert shm_energy("200 N / m", "5 cm").magnitude == pytest.approx(0.25)

    v_mid = shm_speed_at_position("5 rad / s", "10 cm", "-6 cm")
    assert v_mid.magnitude == pytest.approx(5.0 * 0.08)
    assert shm_speed_at_position("5 rad / s", "10 cm", "10 cm").magnitude == 0.0


def test_oscillations_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        spring_mass_angular_frequency("100 N", "1 kg")
    with pytest.raises(pint.DimensionalityError):
        simple_pendulum_period("1 s")
    with pytest.raises(pint.DimensionalityError):
        physical_pendulum_period("1 kg * m", "1 kg", "0.5 m")
    with pytest.raises(pint.DimensionalityError):
        shm_displacement("10 cm", "4 m / s", "1 s")
    with pytest.raises(pint.DimensionalityError):
        shm_displacement("10 cm", "4 rad / s", "1 s", phase="1 m")

    with pytest.raises(PhysicalDomainError):
        spring_mass_angular_frequency("-100 N / m", "1 kg")
    with pytest.raises(PhysicalDomainError):
        spring_mass_angular_frequency("100 N / m", "0 kg")
    with pytest.raises(PhysicalDomainError):
        spring_mass_period("1 kg", "0 N / m")
    with pytest.raises(PhysicalDomainError):
        simple_pendulum_period("-1 m")
    with pytest.raises(PhysicalDomainError):
        simple_pendulum_period("1 m", gravity="0 m/s^2")
    with pytest.raises(PhysicalDomainError):
        physical_pendulum_period("1 kg * m^2", "1 kg", "0 m")
    with pytest.raises(PhysicalDomainError):
        shm_displacement("-5 cm", "4 rad / s", "1 s")
    with pytest.raises(PhysicalDomainError):
        shm_velocity("5 cm", "4 rad / s", "-1 s")
    with pytest.raises(PhysicalDomainError):
        shm_energy("100 N / m", "-2 cm")
    with pytest.raises(PhysicalDomainError):
        shm_speed_at_position("4 rad / s", "10 cm", "12 cm")


def test_wave_kinematics_strings_and_intensity() -> None:
    v = wave_speed("50 cm", "680 Hz")
    assert str(v.units) == "meter / second"
    assert v.magnitude == pytest.approx(340.0)

    lam = wavelength("340 m / s", "1.7 kHz")
    assert str(lam.units) == "meter"
    assert lam.to("cm").magnitude == pytest.approx(20.0)

    freq = wave_frequency("3e8 m / s", "500 nm")
    assert str(freq.units) == "hertz"
    assert freq.to("THz").magnitude == pytest.approx(600.0)

    k = wavenumber("25 cm")
    assert str(k.units) == "radian / meter"
    assert k.magnitude == pytest.approx(8.0 * math.pi)
    assert wave_phase_speed("16*pi rad / s", k).magnitude == pytest.approx(2.0)
    assert wave_phase_speed("20 rad / s", "5 1 / m").magnitude == pytest.approx(4.0)

    v_str = string_wave_speed("90 N", "10 g / m")
    assert str(v_str.units) == "meter / second"
    assert v_str.magnitude == pytest.approx(math.sqrt(90.0 / 0.01))

    p_str = string_wave_power("10 g / m", "50 m / s", "100 rad / s", "2 cm")
    assert str(p_str.units) == "watt"
    assert p_str.magnitude == pytest.approx(0.5 * 0.01 * 50.0 * (100.0**2) * (0.02**2))

    intensity = wave_intensity("60 W", "2 m^2")
    assert str(intensity.units) == "watt / meter ** 2"
    assert intensity.magnitude == pytest.approx(30.0)

    i_sph = spherical_wave_intensity("100 W", "5 m")
    assert i_sph.magnitude == pytest.approx(100.0 / (4.0 * math.pi * 25.0))


def test_mechanical_waves_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        wave_speed("1 m", "2 s")
    with pytest.raises(pint.DimensionalityError):
        string_wave_speed("50 N", "10 kg / m^3")
    with pytest.raises(pint.DimensionalityError):
        wave_phase_speed("10 rad / s", "2 m")
    with pytest.raises(pint.DimensionalityError):
        wave_intensity("100 J", "2 m^2")

    with pytest.raises(PhysicalDomainError):
        wave_speed("-1 m", "10 Hz")
    with pytest.raises(PhysicalDomainError):
        wavelength("340 m / s", "0 Hz")
    with pytest.raises(PhysicalDomainError):
        wave_frequency("340 m / s", "0 m")
    with pytest.raises(PhysicalDomainError):
        wavenumber("-0.5 m")
    with pytest.raises(PhysicalDomainError):
        wave_phase_speed("10 rad / s", "0 1 / m")
    with pytest.raises(PhysicalDomainError):
        string_wave_speed("50 N", "0 kg / m")
    with pytest.raises(PhysicalDomainError):
        string_wave_power("10 g / m", "-5 m / s", "10 rad / s", "1 cm")
    with pytest.raises(PhysicalDomainError):
        wave_intensity("100 W", "0 m^2")
    with pytest.raises(PhysicalDomainError):
        spherical_wave_intensity("100 W", "-2 m")


def test_standing_waves_harmonics_tension_and_beats() -> None:
    # 65 cm guitar string with v = 572 m/s => f_1 = 440 Hz, f_3 = 1320 Hz
    assert string_harmonic_wavelength(1, "65 cm").to("cm").magnitude == pytest.approx(130.0)
    assert string_harmonic_wavelength(2, "65 cm").to("cm").magnitude == pytest.approx(65.0)
    f1 = string_harmonic_frequency(1, "65 cm", "572 m / s")
    assert str(f1.units) == "hertz"
    assert f1.magnitude == pytest.approx(440.0)
    assert open_pipe_harmonic_frequency(3, "65 cm", "572 m / s").magnitude == pytest.approx(1320.0)

    # Closed-open pipe: L = 25 cm, v = 340 m/s => f_1 = 340 Hz, f_3 = 1020 Hz
    assert closed_pipe_harmonic_wavelength(1, "25 cm").magnitude == pytest.approx(1.0)
    assert closed_pipe_harmonic_wavelength(3, "25 cm").magnitude == pytest.approx(1.0 / 3.0)
    assert closed_pipe_harmonic_frequency(1, "25 cm", "340 m / s").magnitude == pytest.approx(340.0)
    assert closed_pipe_harmonic_frequency(3, "25 cm", "340 m / s").magnitude == pytest.approx(
        1020.0
    )

    # Roundtrip string tension -> wave speed -> fundamental frequency
    tension = fundamental_tension_for_frequency("440 Hz", "65 cm", "0.8 g / m")
    assert str(tension.units) == "newton"
    v_check = string_wave_speed(tension, "0.8 g / m")
    assert string_harmonic_frequency(1, "65 cm", v_check).magnitude == pytest.approx(440.0)

    # Beat frequency
    assert beat_frequency("440 Hz", "443.5 Hz").magnitude == pytest.approx(3.5)
    assert beat_frequency("440 Hz", "440 Hz").magnitude == 0.0


def test_standing_waves_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        string_harmonic_frequency("1 m", "1 m", "340 m / s")
    with pytest.raises(pint.DimensionalityError):
        beat_frequency("440 Hz", "2 s")
    with pytest.raises(pint.DimensionalityError):
        fundamental_tension_for_frequency("440 Hz", "0.65 m", "1 kg")

    with pytest.raises(PhysicalDomainError):
        string_harmonic_frequency(0, "1 m", "340 m / s")
    with pytest.raises(PhysicalDomainError):
        string_harmonic_frequency(1.5, "1 m", "340 m / s")
    with pytest.raises(PhysicalDomainError):
        closed_pipe_harmonic_frequency(2, "1 m", "340 m / s")
    with pytest.raises(PhysicalDomainError):
        closed_pipe_harmonic_wavelength(4, "1 m")
    with pytest.raises(PhysicalDomainError):
        string_harmonic_wavelength(1, "0 m")
    with pytest.raises(PhysicalDomainError):
        fundamental_tension_for_frequency("-440 Hz", "0.65 m", "1 g / m")
    with pytest.raises(PhysicalDomainError):
        beat_frequency("-10 Hz", "440 Hz")


def test_acoustics_sound_speed_decibels_and_impedance() -> None:
    r_gas = float(MOLAR_GAS_CONSTANT.to("J / (mol * K)").magnitude)
    # Dry air at 20 degC (293.15 K), gamma = 1.4, M = 28.97 g / mol => ~343.2 m/s
    c_air = speed_of_sound_ideal_gas(1.4, "20 degC", "28.97 g / mol")
    assert str(c_air.units) == "meter / second"
    assert c_air.magnitude == pytest.approx(math.sqrt(1.4 * r_gas * 293.15 / 0.02897))

    beta = sound_intensity_level("1e-6 W / m^2")
    assert str(beta.units) == "decibel"
    assert beta.magnitude == pytest.approx(60.0)
    assert sound_intensity_level("1e-12 W / m^2").magnitude == 0.0

    intensity = intensity_from_sound_level("85 dB")
    assert str(intensity.units) == "watt / meter ** 2"
    assert intensity.magnitude == pytest.approx(1e-12 * (10.0**8.5))
    assert intensity_from_sound_level("1 Np").magnitude == pytest.approx(
        1e-12 * math.exp(2.0), rel=1e-9
    )

    spl = sound_pressure_level("2 Pa")
    assert str(spl.units) == "decibel"
    assert spl.magnitude == pytest.approx(100.0)

    z_air = acoustic_impedance("1.204 kg / m^3", "343 m / s")
    assert str(z_air.units) == "pascal * second / meter"
    assert z_air.magnitude == pytest.approx(1.204 * 343.0)

    z_water = acoustic_impedance("1000 kg / m^3", "1480 m / s")
    r_refl = acoustic_reflection_coefficient(z_air, z_water)
    assert r_refl.dimensionless
    assert r_refl.magnitude == pytest.approx(((1.48e6 - 412.972) / (1.48e6 + 412.972)) ** 2)
    assert acoustic_reflection_coefficient(z_air, z_air).magnitude == 0.0


def test_acoustics_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        speed_of_sound_ideal_gas(1.4, "20 degC", "28 g")
    with pytest.raises(pint.DimensionalityError):
        sound_intensity_level("10 W")
    with pytest.raises(pint.DimensionalityError):
        sound_pressure_level("2 N")
    with pytest.raises(pint.DimensionalityError):
        acoustic_impedance("1.2 kg / m^2", "343 m / s")

    with pytest.raises(PhysicalDomainError):
        speed_of_sound_ideal_gas(0, "20 degC", "28.97 g / mol")
    with pytest.raises(PhysicalDomainError):
        speed_of_sound_ideal_gas(1.4, "-10 K", "28.97 g / mol")
    with pytest.raises(PhysicalDomainError):
        sound_intensity_level("0 W / m^2")
    with pytest.raises(PhysicalDomainError):
        intensity_from_sound_level(60)
    with pytest.raises(PhysicalDomainError):
        intensity_from_sound_level("1 B")
    with pytest.raises(PhysicalDomainError):
        sound_pressure_level("0 Pa")
    with pytest.raises(PhysicalDomainError):
        acoustic_impedance("-1 kg / m^3", "343 m / s")
    with pytest.raises(PhysicalDomainError):
        acoustic_reflection_coefficient("0 Pa * s / m", "0 Pa * s / m")


def test_doppler_and_mach_cone() -> None:
    # Ambulance siren 700 Hz approaching at 34 m/s in 340 m/s air, stationary observer
    f_app = doppler_frequency("700 Hz", "340 m / s", source_velocity="34 m / s")
    assert str(f_app.units) == "hertz"
    assert f_app.magnitude == pytest.approx(700.0 * 340.0 / (340.0 - 34.0))

    # Receding source and moving observer
    f_rec = doppler_frequency(
        "700 Hz",
        "340 m / s",
        observer_velocity="17 m / s",
        source_velocity="-34 m / s",
    )
    assert f_rec.magnitude == pytest.approx(700.0 * (340.0 + 17.0) / (340.0 + 34.0))

    # Relativistic Doppler at beta = 0.6 (approaching => 2x, receding => 0.5x)
    f_blue = relativistic_doppler_frequency("500 THz", "0.6 m / s", speed_of_light="1 m / s")
    f_red = relativistic_doppler_frequency("500 THz", "-0.6 m / s", speed_of_light="1 m / s")
    assert f_blue.to("THz").magnitude == pytest.approx(1000.0)
    assert f_red.to("THz").magnitude == pytest.approx(250.0)

    # Mach 2 jet => Mach angle = 30 deg
    m_val = mach_number("680 m / s", "340 m / s")
    assert m_val.dimensionless
    assert m_val.magnitude == pytest.approx(2.0)

    mu = mach_angle("680 m / s", "340 m / s")
    assert str(mu.units) == "radian"
    assert mu.to("deg").magnitude == pytest.approx(30.0)
    assert mach_angle("340 m / s", "340 m / s").to("deg").magnitude == pytest.approx(90.0)


def test_doppler_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        doppler_frequency("500 Hz", "340 m")
    with pytest.raises(pint.DimensionalityError):
        relativistic_doppler_frequency("500 THz", "0.5 s")
    with pytest.raises(pint.DimensionalityError):
        mach_number("680 m / s", "340 s")

    with pytest.raises(PhysicalDomainError):
        doppler_frequency("500 Hz", "340 m / s", source_velocity="340 m / s")
    with pytest.raises(PhysicalDomainError):
        doppler_frequency("500 Hz", "340 m / s", observer_velocity="-400 m / s")
    with pytest.raises(PhysicalDomainError):
        relativistic_doppler_frequency("500 THz", "1.0 m / s", speed_of_light="1 m / s")
    with pytest.raises(PhysicalDomainError):
        relativistic_doppler_frequency("500 THz", "-1.2 m / s", speed_of_light="1 m / s")
    with pytest.raises(PhysicalDomainError):
        mach_number("-100 m / s", "340 m / s")
    with pytest.raises(PhysicalDomainError):
        mach_number("100 m / s", "0 m / s")
    with pytest.raises(PhysicalDomainError):
        mach_angle("300 m / s", "340 m / s")


def test_waves_external_registry_nonfinite_and_exports() -> None:
    ext_ureg = pint.UnitRegistry()
    v_ext = wave_speed(ext_ureg.Quantity(50, "cm"), ext_ureg.Quantity(2, "kHz"))
    assert v_ext.magnitude == pytest.approx(1000.0)

    # Non-finite rejection
    with pytest.raises(PhysicalDomainError):
        wave_speed("nan m", "10 Hz")
    with pytest.raises(PhysicalDomainError):
        doppler_frequency("500 Hz", "inf m / s")

    # Package __all__ completeness across all 5 submodules
    for submodule in (
        waves.oscillations,
        waves.mechanical,
        waves.standing,
        waves.acoustics,
        waves.doppler,
    ):
        for symbol in submodule.__all__:
            assert symbol in waves.__all__
            assert hasattr(waves, symbol)
