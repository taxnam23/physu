"""Real-world modern physics examples: cosmic muons, photoelectric cell, Bohr, and C-14."""

from __future__ import annotations

from physu.constants import ELECTRON_MASS, PLANCK_CONSTANT
from physu.modern import (
    bohr_energy_level,
    bohr_orbit_radius,
    compton_wavelength_shift,
    de_broglie_wavelength_particle,
    heisenberg_min_energy_uncertainty,
    infinite_well_energy,
    length_contraction,
    mass_energy_equivalence,
    photoelectric_max_kinetic_energy,
    photoelectric_stopping_potential,
    photon_energy,
    radioactive_activity,
    radiometric_age,
    relativistic_kinetic_energy,
    rydberg_wavelength,
    time_dilation,
    wien_peak_wavelength,
)


def main() -> None:
    """Solve practical relativity, photon, quantum, atomic, and nuclear physics problems."""
    # 1. Cosmic-ray muon survival through Earth's atmosphere & relativistic electron energy
    muon_proper_life = "2.2 us"
    muon_lab_life = time_dilation(muon_proper_life, "0.995 c")
    atmos_contracted = length_contraction("10 km", "0.995 c")
    e_kin_fast = relativistic_kinetic_energy(ELECTRON_MASS, "0.9 c")

    print("=== Special Relativity: Cosmic Muons & Relativistic Electron ===")
    print(
        f"Muon lifetime @ 0.995 c: {muon_lab_life.to('us'):.1f} | "
        f"10 km atmosphere contracts to {atmos_contracted.to('km'):.2f}"
    )
    print(f"Electron kinetic energy @ 0.9 c: {e_kin_fast.to('MeV'):.3f}")

    # 2. Cesium photocell, X-ray Compton scattering, and solar Wien peak
    e_uv = photon_energy(wavelength="400 nm")
    k_photo = photoelectric_max_kinetic_energy(e_uv / PLANCK_CONSTANT, "2.14 eV")
    v_stop = photoelectric_stopping_potential(k_photo)
    dlam_compton = compton_wavelength_shift("90 deg")
    lam_sun = wien_peak_wavelength("5778 K")

    print("\n=== Photons, Photoelectric Effect & Wien's Law ===")
    print(f"400 nm photon: {e_uv.to('eV'):.2f} | Cs stopping voltage: {v_stop:.2f}")
    print(
        f"90 deg Compton shift: {dlam_compton.to('pm'):.3f} | "
        f"5778 K blackbody peak: {lam_sun.to('nm'):.1f}"
    )

    # 3. Electron diffraction, quantum dot well, and hydrogen Balmer-alpha line
    lam_electron = de_broglie_wavelength_particle(ELECTRON_MASS, "5e6 m / s")
    de_lifetime = heisenberg_min_energy_uncertainty("10 ns")
    e_well_1 = infinite_well_energy(1, ELECTRON_MASS, "2 nm")
    r_bohr_1 = bohr_orbit_radius(1)
    e_h_1 = bohr_energy_level(1)
    lam_balmer = rydberg_wavelength(3, 2)

    print("\n=== Quantum Mechanics & Bohr Atom ===")
    print(f"5e6 m/s electron de Broglie wavelength: {lam_electron.to('angstrom'):.2f}")
    print(f"10 ns state Heisenberg width: {de_lifetime.to('neV'):.2f}")
    print(f"2 nm 1D quantum well ground state: {e_well_1.to('meV'):.1f}")
    print(
        f"Hydrogen n=1: r1 = {r_bohr_1.to('angstrom'):.3f}, "
        f"E1 = {e_h_1.to('eV'):.2f} | H-alpha = {lam_balmer.to('nm'):.1f}"
    )

    # 4. Nuclear binding energy scale and Carbon-14 archaeological dating
    e_amu = mass_energy_equivalence("1 u")
    act_c14 = radioactive_activity(1.0e18, half_life="5730 year")
    sample_age = radiometric_age("1.0 g", "0.35 g", half_life="5730 year")

    print("\n=== Nuclear Physics & Radiocarbon Dating ===")
    print(f"1 atomic mass unit equivalence: {e_amu.to('MeV'):.3f}")
    print(f"1e18 C-14 nuclei activity: {act_c14.to('MBq'):.2f}")
    print(f"Sample with 35% C-14 remaining age: {sample_age.to('year'):.0f}")


if __name__ == "__main__":
    main()
