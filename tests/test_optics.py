"""Unit tests for physu.optics submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError
from physu.constants import SPEED_OF_LIGHT
from physu.optics import (
    brewster_angle,
    critical_angle,
    refractive_index,
    snell_refraction_angle,
    speed_in_medium,
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
