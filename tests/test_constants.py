"""Unit tests for physu.constants."""

from __future__ import annotations

import math

import pytest

from physu import Quantity, constants


def test_constants_bound_to_central_registry() -> None:
    for name in constants.__all__:
        value = getattr(constants, name)
        assert isinstance(value, Quantity), f"{name} must belong to physu.ureg"


def test_fundamental_constant_values_and_dimensions() -> None:
    assert constants.STANDARD_GRAVITY.to("m / s^2").magnitude == pytest.approx(9.80665)
    assert constants.SPEED_OF_LIGHT.to("m / s").magnitude == pytest.approx(299_792_458)
    assert constants.GRAVITATIONAL_CONSTANT.to("m^3 / (kg * s^2)").magnitude == pytest.approx(
        6.67430e-11
    )
    assert constants.PLANCK_CONSTANT.to("J * s").magnitude == pytest.approx(6.62607015e-34)
    assert constants.REDUCED_PLANCK_CONSTANT.to("J * s").magnitude == pytest.approx(
        6.62607015e-34 / (2.0 * math.pi)
    )
    assert constants.ELEMENTARY_CHARGE.to("C").magnitude == pytest.approx(1.602176634e-19)
    assert constants.BOLTZMANN_CONSTANT.to("J / K").magnitude == pytest.approx(1.380649e-23)
    assert constants.AVOGADRO_CONSTANT.to("1 / mol").magnitude == pytest.approx(6.02214076e23)
    assert constants.MOLAR_GAS_CONSTANT.to("J / (mol * K)").magnitude == pytest.approx(
        8.31446261815324
    )
    assert constants.STANDARD_ATMOSPHERE.to("Pa").magnitude == pytest.approx(101_325)
    assert constants.STEFAN_BOLTZMANN_CONSTANT.to("W / (m^2 * K^4)").magnitude == pytest.approx(
        5.670374419e-8, rel=1e-9
    )
