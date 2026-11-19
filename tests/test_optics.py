"""Unit tests for physu.optics submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import SPEED_OF_LIGHT
from physu.optics import (
    bragg_diffraction_angle,
    brewster_angle,
    combined_optical_power,
    critical_angle,
    diffraction_grating_angle,
    double_slit_angle,
    double_slit_fringe_position,
    double_slit_fringe_spacing,
    grating_spacing_from_density,
    lateral_magnification,
    lensmaker_focal_length,
    malus_intensity,
    michelson_displacement,
    mirror_focal_length,
    optical_power,
    polarization_degree,
    polarizer_stack_intensity,
    rayleigh_angular_resolution,
    refractive_index,
    single_slit_minimum_angle,
    snell_refraction_angle,
    speed_in_medium,
    thin_film_constructive_thickness,
    thin_lens_focal_length,
    thin_lens_image_distance,
    unpolarized_polarizer_intensity,
    wavelength_in_medium,
)


def test_refraction_media_and_angles() -> None:
    c_val = float(SPEED_OF_LIGHT.to("m / s").magnitude)

    v_glass = speed_in_medium(1.5)
    assert str(v_glass.units) == "meter / second"
    assert v_glass.magnitude == pytest.approx(c_val / 1.5)

    n_glass = refractive_index(v_glass)
    assert n_glass.dimensionless
    assert n_glass.magnitude == pytest.approx(1.5)

    lam_water = wavelength_in_medium("532 nm", 1.333)
    assert str(lam_water.units) == "meter"
    assert lam_water.to("nm").magnitude == pytest.approx(532.0 / 1.333)

    # Air (1.0) -> Crown glass (1.5) at 30 deg => arcsin(0.5 / 1.5)
    theta_2 = snell_refraction_angle("30 deg", 1.0, 1.5)
    assert str(theta_2.units) == "radian"
    assert theta_2.to("deg").magnitude == pytest.approx(math.degrees(math.asin(1.0 / 3.0)))
    assert snell_refraction_angle("0 deg", 1.0, 1.5).magnitude == 0.0

    # Glass (1.5) -> Air (1.0) critical angle & Brewster angle
    theta_c = critical_angle(1.5, 1.0)
    assert theta_c.to("deg").magnitude == pytest.approx(math.degrees(math.asin(1.0 / 1.5)))
    assert critical_angle(1.5, 1.5).to("deg").magnitude == pytest.approx(90.0)

    theta_b = brewster_angle(1.0, 1.5)
    assert theta_b.to("deg").magnitude == pytest.approx(math.degrees(math.atan(1.5)))


def test_refraction_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        speed_in_medium("1.5 m")
    with pytest.raises(pint.DimensionalityError):
        wavelength_in_medium("532 Hz", 1.5)
    with pytest.raises(pint.DimensionalityError):
        refractive_index("2e8 m")
    with pytest.raises(pint.DimensionalityError):
        snell_refraction_angle("30 m", 1.0, 1.5)

    with pytest.raises(PhysicalDomainError):
        speed_in_medium(0)
    with pytest.raises(PhysicalDomainError):
        speed_in_medium(-1.5)
    with pytest.raises(PhysicalDomainError):
        wavelength_in_medium("-500 nm", 1.5)
    with pytest.raises(PhysicalDomainError):
        refractive_index("0 m / s")
    with pytest.raises(PhysicalDomainError):
        snell_refraction_angle("-5 deg", 1.0, 1.5)
    with pytest.raises(PhysicalDomainError):
        snell_refraction_angle("95 deg", 1.0, 1.5)
    with pytest.raises(PhysicalDomainError):
        snell_refraction_angle("45 deg", 1.5, 1.0)
    with pytest.raises(PhysicalDomainError):
        critical_angle(1.0, 1.5)
    with pytest.raises(PhysicalDomainError):
        brewster_angle(0, 1.5)


def test_lenses_mirrors_and_optical_power() -> None:
    # Converging lens f = 20 cm, object at 30 cm => image at 60 cm, M = -2
    d_i = thin_lens_image_distance("20 cm", "30 cm")
    assert str(d_i.units) == "meter"
    assert d_i.to("cm").magnitude == pytest.approx(60.0)

    f_check = thin_lens_focal_length("30 cm", d_i)
    assert f_check.to("cm").magnitude == pytest.approx(20.0)

    mag = lateral_magnification("30 cm", d_i)
    assert mag.dimensionless
    assert mag.magnitude == pytest.approx(-2.0)

    # Diverging lens f = -25 cm => power = -4.0 / m
    p_div = optical_power("-25 cm")
    assert str(p_div.units) == "1 / meter"
    assert p_div.magnitude == pytest.approx(-4.0)

    p_comb = combined_optical_power(["20 cm", "-25 cm"])
    assert str(p_comb.units) == "1 / meter"
    assert p_comb.magnitude == pytest.approx(1.0)

    # Symmetric biconvex lens n = 1.5, R1 = +20 cm, R2 = -20 cm => f = 20 cm
    f_maker = lensmaker_focal_length(1.5, "20 cm", "-20 cm")
    assert f_maker.to("cm").magnitude == pytest.approx(20.0)

    assert mirror_focal_length("-40 cm").to("cm").magnitude == pytest.approx(-20.0)


def test_lenses_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        thin_lens_image_distance("20 cm", "30 s")
    with pytest.raises(pint.DimensionalityError):
        optical_power("5 s")
    with pytest.raises(pint.DimensionalityError):
        combined_optical_power(["20 cm", "10 s"])

    with pytest.raises(TypeError):
        combined_optical_power("20 cm")

    with pytest.raises(PhysicalDomainError):
        thin_lens_image_distance("20 cm", "20 cm")
    with pytest.raises(PhysicalDomainError):
        thin_lens_image_distance("0 cm", "30 cm")
    with pytest.raises(PhysicalDomainError):
        thin_lens_focal_length("20 cm", "-20 cm")
    with pytest.raises(PhysicalDomainError):
        lateral_magnification("0 cm", "20 cm")
    with pytest.raises(PhysicalDomainError):
        optical_power("0 m")
    with pytest.raises(PhysicalDomainError):
        combined_optical_power([])
    with pytest.raises(PhysicalDomainError):
        lensmaker_focal_length(1.5, "20 cm", "-20 cm", surrounding_index=1.5)
    with pytest.raises(PhysicalDomainError):
        lensmaker_focal_length(1.5, "20 cm", "20 cm")
    with pytest.raises(PhysicalDomainError):
        lensmaker_focal_length(1.5, "0 cm", "-20 cm")


def test_interference_double_slit_thin_film_and_michelson() -> None:
    theta_1 = double_slit_angle(1, "500 nm", "1 um")
    assert str(theta_1.units) == "radian"
    assert theta_1.to("deg").magnitude == pytest.approx(30.0)
    assert double_slit_angle(0, "500 nm", "1 um").magnitude == 0.0

    dy = double_slit_fringe_spacing("600 nm", "2 m", "0.3 mm")
    assert str(dy.units) == "meter"
    assert dy.to("mm").magnitude == pytest.approx(4.0)
    assert double_slit_fringe_position(3, "600 nm", "2 m", "0.3 mm").to("mm").magnitude == (
        pytest.approx(12.0)
    )

    # Soap film (n = 1.33) in air (1 phase shift), m = 0, lambda = 532 nm => 100 nm
    t_soap = thin_film_constructive_thickness("532 nm", 1.33, order=0, phase_shifts=1)
    assert t_soap.to("nm").magnitude == pytest.approx(100.0)
    t_two = thin_film_constructive_thickness("532 nm", 1.33, order=1, phase_shifts=2)
    assert t_two.to("nm").magnitude == pytest.approx(200.0)

    disp = michelson_displacement(400, "632.8 nm")
    assert disp.to("um").magnitude == pytest.approx(126.56)


def test_interference_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        double_slit_angle(1, "500 Hz", "1 um")
    with pytest.raises(pint.DimensionalityError):
        michelson_displacement("10 m", "632.8 nm")

    with pytest.raises(PhysicalDomainError):
        double_slit_angle(3, "500 nm", "1 um")
    with pytest.raises(PhysicalDomainError):
        double_slit_angle(1.5, "500 nm", "1 um")
    with pytest.raises(PhysicalDomainError):
        double_slit_fringe_position(1, "500 nm", "2 m", "0 mm")
    with pytest.raises(PhysicalDomainError):
        thin_film_constructive_thickness("532 nm", 1.33, phase_shifts=3)
    with pytest.raises(PhysicalDomainError):
        thin_film_constructive_thickness("532 nm", 0)
    with pytest.raises(PhysicalDomainError):
        michelson_displacement(-10, "632.8 nm")


def test_diffraction_slit_grating_rayleigh_and_bragg() -> None:
    theta_min = single_slit_minimum_angle(1, "500 nm", "1 um")
    assert str(theta_min.units) == "radian"
    assert theta_min.to("deg").magnitude == pytest.approx(30.0)

    d_grating = grating_spacing_from_density("500 / mm")
    assert str(d_grating.units) == "meter"
    assert d_grating.to("um").magnitude == pytest.approx(2.0)

    theta_gr = diffraction_grating_angle(1, "500 nm", d_grating)
    assert theta_gr.to("deg").magnitude == pytest.approx(math.degrees(math.asin(0.25)))

    theta_r = rayleigh_angular_resolution("550 nm", "2.4 m")
    assert str(theta_r.units) == "radian"
    assert theta_r.to("arcsec").magnitude == pytest.approx(
        math.degrees(1.22 * 550e-9 / 2.4) * 3600.0
    )

    theta_bragg = bragg_diffraction_angle(1, "0.15 nm", "0.3 nm")
    assert theta_bragg.to("deg").magnitude == pytest.approx(math.degrees(math.asin(0.25)))


def test_diffraction_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        grating_spacing_from_density("500 mm")
    with pytest.raises(pint.DimensionalityError):
        rayleigh_angular_resolution("550 Hz", "2.4 m")

    with pytest.raises(PhysicalDomainError):
        single_slit_minimum_angle(0, "500 nm", "1 um")
    with pytest.raises(PhysicalDomainError):
        single_slit_minimum_angle(3, "500 nm", "1 um")
    with pytest.raises(PhysicalDomainError):
        grating_spacing_from_density("0 / mm")
    with pytest.raises(PhysicalDomainError):
        rayleigh_angular_resolution("550 nm", "0 m")
    with pytest.raises(PhysicalDomainError):
        bragg_diffraction_angle(0, "0.15 nm", "0.3 nm")
    with pytest.raises(PhysicalDomainError):
        bragg_diffraction_angle(5, "0.15 nm", "0.3 nm")


def test_polarization_malus_stacks_and_degree() -> None:
    i_half = unpolarized_polarizer_intensity("100 W / m^2")
    assert str(i_half.units) == "watt / meter ** 2"
    assert i_half.magnitude == pytest.approx(50.0)

    assert malus_intensity("100 W / m^2", "60 deg").magnitude == pytest.approx(25.0)
    assert malus_intensity("100 W / m^2", "90 deg").magnitude == 0.0

    # Three-polarizer paradox: 45 deg then 45 deg gives I0 * 0.5 * 0.5 = 0.25 I0
    i_stack = polarizer_stack_intensity("100 W / m^2", ["45 deg", "45 deg"])
    assert i_stack.magnitude == pytest.approx(25.0)

    p_deg = polarization_degree("80 W / m^2", "20 W / m^2")
    assert p_deg.dimensionless
    assert p_deg.magnitude == pytest.approx(0.6)
    assert polarization_degree("50 W / m^2", "50 W / m^2").magnitude == 0.0


def test_polarization_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        malus_intensity("100 W", "30 deg")
    with pytest.raises(pint.DimensionalityError):
        malus_intensity("100 W / m^2", "30 m")
    with pytest.raises(pint.DimensionalityError):
        polarizer_stack_intensity("100 W / m^2", ["45 deg", "10 s"])

    with pytest.raises(TypeError):
        polarizer_stack_intensity("100 W / m^2", "45 deg")

    with pytest.raises(PhysicalDomainError):
        unpolarized_polarizer_intensity("-10 W / m^2")
    with pytest.raises(PhysicalDomainError):
        polarizer_stack_intensity("100 W / m^2", [])
    with pytest.raises(PhysicalDomainError):
        polarization_degree("0 W / m^2", "0 W / m^2")
    with pytest.raises(PhysicalDomainError):
        polarization_degree("20 W / m^2", "40 W / m^2")
