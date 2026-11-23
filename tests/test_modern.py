"""Unit tests for physu.modern submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import (
    BOHR_RADIUS,
    ELECTRON_MASS,
    PLANCK_CONSTANT,
    REDUCED_PLANCK_CONSTANT,
    RYDBERG_CONSTANT,
    SPEED_OF_LIGHT,
    WIEN_DISPLACEMENT_CONSTANT,
)
from physu.modern import (
    bohr_energy_level,
    bohr_orbit_radius,
    bohr_transition_energy,
    compton_wavelength_shift,
    de_broglie_wavelength,
    de_broglie_wavelength_particle,
    decay_constant,
    energy_momentum_relation,
    half_life_from_decay_constant,
    heisenberg_min_energy_uncertainty,
    heisenberg_min_momentum_uncertainty,
    infinite_well_energy,
    length_contraction,
    lorentz_factor,
    mass_energy_equivalence,
    photoelectric_max_kinetic_energy,
    photoelectric_stopping_potential,
    photoelectric_threshold_frequency,
    photon_energy,
    photon_momentum,
    radioactive_activity,
    radioactive_remaining,
    radiometric_age,
    relativistic_kinetic_energy,
    relativistic_momentum,
    relativistic_total_energy,
    relativistic_velocity_addition,
    rest_energy,
    rydberg_wavelength,
    time_dilation,
    wien_peak_wavelength,
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


def test_photons_photoelectric_compton_and_wien() -> None:
    h_val = float(PLANCK_CONSTANT.to("J * s").magnitude)
    c_val = float(SPEED_OF_LIGHT.to("m / s").magnitude)

    e_uv = photon_energy(wavelength="200 nm")
    assert str(e_uv.units) == "joule"
    assert e_uv.to("eV").magnitude == pytest.approx((h_val * c_val / 200e-9) / 1.602176634e-19)

    e_freq = photon_energy(frequency="500 THz")
    p_freq = photon_momentum(frequency="500 THz")
    assert str(p_freq.units) == "kilogram * meter / second"
    assert p_freq.magnitude == pytest.approx(e_freq.magnitude / c_val)

    # Potassium work function = 2.30 eV, incident 400 nm UV photon (~3.10 eV)
    f_inc = c_val / 400e-9
    k_max = photoelectric_max_kinetic_energy(f"{f_inc} Hz", "2.30 eV")
    assert k_max.to("eV").magnitude == pytest.approx(
        photon_energy(wavelength="400 nm").to("eV").magnitude - 2.30
    )
    v_stop = photoelectric_stopping_potential(k_max)
    assert str(v_stop.units) == "volt"
    assert v_stop.magnitude == pytest.approx(k_max.to("eV").magnitude)

    f0 = photoelectric_threshold_frequency("2.30 eV")
    assert str(f0.units) == "hertz"
    assert photoelectric_max_kinetic_energy(f0, "2.30 eV").magnitude == 0.0

    # Compton scattering at 90 deg => lambda_C ; 0 deg => 0 ; 180 deg => 2 * lambda_C
    m_e = float(ELECTRON_MASS.to("kg").magnitude)
    lam_c = h_val / (m_e * c_val)
    assert compton_wavelength_shift("90 deg").magnitude == pytest.approx(lam_c)
    assert compton_wavelength_shift("180 deg").magnitude == pytest.approx(2.0 * lam_c)
    assert compton_wavelength_shift("0 deg").magnitude == 0.0

    # Sun surface ~5778 K => peak ~501.5 nm
    lam_sun = wien_peak_wavelength("5778 K")
    b_val = float(WIEN_DISPLACEMENT_CONSTANT.to("m * K").magnitude)
    assert lam_sun.to("nm").magnitude == pytest.approx((b_val / 5778.0) * 1e9)


def test_photons_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        photon_energy(wavelength="500 Hz")
    with pytest.raises(pint.DimensionalityError):
        photoelectric_threshold_frequency("2.3 V")
    with pytest.raises(pint.DimensionalityError):
        compton_wavelength_shift("90 m")
    with pytest.raises(pint.DimensionalityError):
        wien_peak_wavelength("300 m")

    with pytest.raises(ValueError):
        photon_energy()
    with pytest.raises(ValueError):
        photon_energy(frequency="500 THz", wavelength="600 nm")

    with pytest.raises(PhysicalDomainError):
        photon_energy(wavelength="0 nm")
    with pytest.raises(PhysicalDomainError):
        photoelectric_max_kinetic_energy("400 THz", "3.0 eV")
    with pytest.raises(PhysicalDomainError):
        photoelectric_stopping_potential("-1 eV")
    with pytest.raises(PhysicalDomainError):
        compton_wavelength_shift("90 deg", particle_mass="0 kg")
    with pytest.raises(PhysicalDomainError):
        wien_peak_wavelength("0 K")
    with pytest.raises(PhysicalDomainError):
        wien_peak_wavelength("-300 degC")


def test_quantum_de_broglie_heisenberg_and_well() -> None:
    h_val = float(PLANCK_CONSTANT.to("J * s").magnitude)
    hbar_val = float(REDUCED_PLANCK_CONSTANT.to("J * s").magnitude)
    m_e = float(ELECTRON_MASS.to("kg").magnitude)

    lam_p = de_broglie_wavelength("1e-24 kg * m / s")
    assert str(lam_p.units) == "meter"
    assert lam_p.magnitude == pytest.approx(h_val / 1e-24)

    # Non-relativistic vs relativistic electron at 0.6 c (gamma = 1.25)
    lam_nr = de_broglie_wavelength_particle(ELECTRON_MASS, "0.6 c", relativistic=False)
    lam_rel = de_broglie_wavelength_particle(ELECTRON_MASS, "0.6 c", relativistic=True)
    assert lam_rel.magnitude == pytest.approx(lam_nr.magnitude / 1.25)

    dp = heisenberg_min_momentum_uncertainty("1 angstrom")
    assert str(dp.units) == "kilogram * meter / second"
    assert dp.magnitude == pytest.approx(hbar_val / (2.0 * 1e-10))

    de = heisenberg_min_energy_uncertainty("1 ns")
    assert str(de.units) == "joule"
    assert de.magnitude == pytest.approx(hbar_val / (2.0 * 1e-9))

    e1 = infinite_well_energy(1, ELECTRON_MASS, "1 nm")
    e3 = infinite_well_energy(3, ELECTRON_MASS, "1 nm")
    assert str(e1.units) == "joule"
    assert e1.magnitude == pytest.approx((h_val**2) / (8.0 * m_e * (1e-9**2)))
    assert e3.magnitude == pytest.approx(9.0 * e1.magnitude)


def test_quantum_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        de_broglie_wavelength("10 J")
    with pytest.raises(pint.DimensionalityError):
        heisenberg_min_momentum_uncertainty("1 s")
    with pytest.raises(pint.DimensionalityError):
        heisenberg_min_energy_uncertainty("1 m")
    with pytest.raises(pint.DimensionalityError):
        infinite_well_energy(1, ELECTRON_MASS, "1 s")

    with pytest.raises(PhysicalDomainError):
        de_broglie_wavelength("0 kg * m / s")
    with pytest.raises(PhysicalDomainError):
        de_broglie_wavelength_particle("0 kg", "1e5 m / s")
    with pytest.raises(PhysicalDomainError):
        de_broglie_wavelength_particle(ELECTRON_MASS, "0 m / s")
    with pytest.raises(PhysicalDomainError):
        de_broglie_wavelength_particle(ELECTRON_MASS, "1.0 c", relativistic=True)
    with pytest.raises(PhysicalDomainError):
        heisenberg_min_momentum_uncertainty("0 m")
    with pytest.raises(PhysicalDomainError):
        heisenberg_min_energy_uncertainty("-1 ns")
    with pytest.raises(PhysicalDomainError):
        infinite_well_energy(0, ELECTRON_MASS, "1 nm")
    with pytest.raises(PhysicalDomainError):
        infinite_well_energy(1.5, ELECTRON_MASS, "1 nm")
    with pytest.raises(PhysicalDomainError):
        infinite_well_energy(1, ELECTRON_MASS, "0 nm")


def test_atomic_bohr_and_rydberg() -> None:
    a0 = float(BOHR_RADIUS.to("m").magnitude)
    r_inf = float(RYDBERG_CONSTANT.to("1 / m").magnitude)

    assert bohr_orbit_radius(1).magnitude == pytest.approx(a0)
    assert bohr_orbit_radius(2, atomic_number=2).magnitude == pytest.approx(2.0 * a0)

    e1 = bohr_energy_level(1)
    assert str(e1.units) == "joule"
    assert e1.to("eV").magnitude == pytest.approx(-13.605693, rel=1e-6)
    assert bohr_energy_level(2, atomic_number=2).to("eV").magnitude == pytest.approx(
        e1.to("eV").magnitude
    )

    # Balmer alpha emission n=3 -> n=2 (~ -1.89 eV, ~656.1 nm)
    de_balmer = bohr_transition_energy(3, 2)
    assert de_balmer.to("eV").magnitude == pytest.approx(
        -13.605693 * (1.0 / 4.0 - 1.0 / 9.0), rel=1e-6
    )
    assert bohr_transition_energy(2, 2).magnitude == 0.0

    lam_alpha = rydberg_wavelength(3, 2)
    assert str(lam_alpha.units) == "meter"
    assert lam_alpha.magnitude == pytest.approx(1.0 / (r_inf * (1.0 / 4.0 - 1.0 / 9.0)))
    assert photon_energy(wavelength=lam_alpha).magnitude == pytest.approx(-de_balmer.magnitude)


def test_atomic_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        bohr_orbit_radius(1, bohr_radius="1 s")
    with pytest.raises(pint.DimensionalityError):
        rydberg_wavelength(3, 2, rydberg_constant="1 m")

    with pytest.raises(PhysicalDomainError):
        bohr_orbit_radius(0)
    with pytest.raises(PhysicalDomainError):
        bohr_orbit_radius(1, atomic_number=0)
    with pytest.raises(PhysicalDomainError):
        bohr_energy_level(1.5)
    with pytest.raises(PhysicalDomainError):
        rydberg_wavelength(2, 2)
    with pytest.raises(PhysicalDomainError):
        rydberg_wavelength(3, 2, rydberg_constant="0 / m")


def test_nuclear_binding_energy_and_decay() -> None:
    # 1 u = 931.494 MeV
    e_u = mass_energy_equivalence("1 u")
    assert str(e_u.units) == "joule"
    assert e_u.to("MeV").magnitude == pytest.approx(931.494, rel=1e-4)

    # Carbon-14 half-life = 5730 years
    lam_c14 = decay_constant("5730 year")
    assert str(lam_c14.units) == "1 / second"
    assert half_life_from_decay_constant(lam_c14).to("year").magnitude == pytest.approx(5730.0)

    # After 2 half-lives (11460 years), 100 g decays to 25 g
    rem = radioactive_remaining("100 g", "11460 year", half_life="5730 year")
    assert str(rem.units) == "gram"
    assert rem.magnitude == pytest.approx(25.0)
    assert radioactive_remaining("100 g", "0 s", decay_constant=lam_c14).magnitude == pytest.approx(
        100.0
    )

    # Activity in Bq and Ci
    act = radioactive_activity(1e20, half_life="1000 s")
    assert str(act.units) == "becquerel"
    assert act.magnitude == pytest.approx(1e20 * math.log(2.0) / 1000.0)
    assert act.to("Ci").magnitude == pytest.approx(act.magnitude / 3.7e10, rel=1e-6)

    age = radiometric_age("100 g", "25 g", half_life="5730 year")
    assert str(age.units) == "second"
    assert age.to("year").magnitude == pytest.approx(11460.0)
    assert radiometric_age("100 g", "100 g", half_life="5730 year").magnitude == 0.0


def test_nuclear_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        mass_energy_equivalence("1 J")
    with pytest.raises(pint.DimensionalityError):
        decay_constant("10 m")
    with pytest.raises(pint.DimensionalityError):
        radiometric_age("100 g", "25 s", half_life="5730 year")

    with pytest.raises(ValueError):
        radioactive_remaining("100 g", "10 s")
    with pytest.raises(ValueError):
        radioactive_activity(1e10, half_life="10 s", decay_constant="0.1 / s")

    with pytest.raises(PhysicalDomainError):
        mass_energy_equivalence("-1 u")
    with pytest.raises(PhysicalDomainError):
        decay_constant("0 s")
    with pytest.raises(PhysicalDomainError):
        half_life_from_decay_constant("0 / s")
    with pytest.raises(PhysicalDomainError):
        radioactive_remaining("-10 g", "10 s", half_life="5 s")
    with pytest.raises(PhysicalDomainError):
        radioactive_remaining("10 g", "-1 s", half_life="5 s")
    with pytest.raises(PhysicalDomainError):
        radiometric_age("10 g", "20 g", half_life="5730 year")
    with pytest.raises(PhysicalDomainError):
        radiometric_age("10 g", "0 g", half_life="5730 year")
