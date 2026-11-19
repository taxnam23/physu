"""Real-world optics examples: fiber optics, camera lenses, AR coating, telescope, and XRD."""

from __future__ import annotations

from physu.optics import (
    bragg_diffraction_angle,
    brewster_angle,
    combined_optical_power,
    critical_angle,
    diffraction_grating_angle,
    double_slit_fringe_spacing,
    grating_spacing_from_density,
    lateral_magnification,
    michelson_displacement,
    optical_power,
    polarizer_stack_intensity,
    rayleigh_angular_resolution,
    speed_in_medium,
    thin_film_constructive_thickness,
    thin_lens_image_distance,
    unpolarized_polarizer_intensity,
    wavelength_in_medium,
)


def main() -> None:
    """Solve practical geometric and wave optics engineering problems."""
    # 1. Optical fiber core/cladding, water Brewster glare, and polarizer stack
    fiber_theta_c = critical_angle(1.48, 1.46)
    v_core = speed_in_medium(1.48)
    lam_core = wavelength_in_medium("1550 nm", 1.48)
    theta_b_water = brewster_angle(1.0, 1.333)
    i_sunglass = unpolarized_polarizer_intensity("800 W / m^2")
    i_stack = polarizer_stack_intensity(i_sunglass, ["30 deg", "45 deg"])

    print("=== Refraction & Polarization: Fiber, Brewster & Analyzers ===")
    print(f"Fiber critical angle (1.48 -> 1.46): {fiber_theta_c.to('deg'):.2f}")
    print(
        f"1550 nm telecom laser in core: v = {v_core.to('km / s'):.0f}, "
        f"lam = {lam_core.to('nm'):.1f}"
    )
    print(f"Water Brewster angle: {theta_b_water.to('deg'):.1f}")
    print(f"800 W/m^2 unpolarized -> 1st polarizer: {i_sunglass:.0f} -> stack: {i_stack:.1f}")

    # 2. Projector lens imaging & combined optometry prescription
    d_image = thin_lens_image_distance("50 mm", "52 mm")
    mag = lateral_magnification("52 mm", d_image)
    p_50mm = optical_power("50 mm")
    p_combo = combined_optical_power(["50 cm", "-25 cm"])

    print("\n=== Thin Lenses & Optical Power ===")
    print(f"50 mm lens @ 52 mm object: image at {d_image:.2f}, magnification = {mag:.1f}")
    print(f"50 mm lens optical power: {p_50mm:.1f} | Combined (50 cm + -25 cm): {p_combo:.1f}")

    # 3. Thin-film soap/coating reflection, double-slit laser, and Michelson shift
    t_coat = thin_film_constructive_thickness("550 nm", 1.38, order=0, phase_shifts=1)
    dy_slits = double_slit_fringe_spacing("632.8 nm", "2.5 m", "0.25 mm")
    michelson_d = michelson_displacement(250, "632.8 nm")

    print("\n=== Interference: Thin Film, Double Slit & Michelson ===")
    print(f"Min constructive thickness (550 nm, n=1.38): {t_coat.to('nm'):.1f}")
    print(f"HeNe 632.8 nm fringe spacing (L=2.5 m, d=0.25 mm): {dy_slits.to('mm'):.2f}")
    print(f"Michelson mirror shift (250 fringes @ 632.8 nm): {michelson_d.to('um'):.2f}")

    # 4. Spectrometer grating, 2.4 m space telescope Rayleigh limit, and NaCl XRD
    d_grating = grating_spacing_from_density("600 / mm")
    theta_na = diffraction_grating_angle(1, "589 nm", d_grating)
    theta_hubble = rayleigh_angular_resolution("550 nm", "2.4 m")
    theta_xrd = bragg_diffraction_angle(1, "0.154 nm", "0.282 nm")

    print("\n=== Diffraction: Spectrometer, Telescope & X-Ray Bragg ===")
    print(f"600 lines/mm 1st-order Na 589 nm angle: {theta_na.to('deg'):.2f}")
    print(f"2.4 m aperture Rayleigh limit @ 550 nm: {theta_hubble.to('arcsec'):.3f}")
    print(f"Cu K-alpha 0.154 nm Bragg angle on NaCl: {theta_xrd.to('deg'):.2f}")


if __name__ == "__main__":
    main()
