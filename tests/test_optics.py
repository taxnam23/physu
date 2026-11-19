"""Unit tests for physu.optics submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import SPEED_OF_LIGHT
from physu.optics import (
    brewster_angle,
    combined_optical_power,
    critical_angle,
    lateral_magnification,
    lensmaker_focal_length,
    mirror_focal_length,
    optical_power,
    refractive_index,
    snell_refraction_angle,
    speed_in_medium,
    thin_lens_focal_length,
    thin_lens_image_distance,
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
