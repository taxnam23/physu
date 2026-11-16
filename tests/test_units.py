"""Unit tests for physu.units."""

from __future__ import annotations

import pint
import pytest

import physu
from physu import Quantity, ureg
from physu.units import check_dimension, convert_to, to_quantity


def test_centralized_registry() -> None:
    assert physu.ureg is physu.units.ureg
    q = 5 * ureg.meter
    assert isinstance(q, Quantity)


def test_to_quantity_normalizes_supported_inputs() -> None:
    q_local = 10 * ureg.meter
    assert to_quantity(q_local) is q_local

    other_ureg = pint.UnitRegistry()
    q_other = 36 * (other_ureg.kilometer / other_ureg.hour)
    normalized = to_quantity(q_other)
    assert isinstance(normalized, Quantity)
    assert normalized.to("m / s").magnitude == pytest.approx(10.0)

    q_str = to_quantity("2.5 min")
    assert isinstance(q_str, Quantity)
    assert q_str.to("s").magnitude == pytest.approx(150.0)

    q_scalar = to_quantity(3.5)
    assert q_scalar.dimensionless
    assert q_scalar.magnitude == pytest.approx(3.5)


@pytest.mark.parametrize("bad_value", [True, False, [1, 2], None])
def test_to_quantity_rejects_invalid_types(bad_value: object) -> None:
    with pytest.raises(TypeError):
        to_quantity(bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
def test_to_quantity_rejects_empty_strings(empty: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        to_quantity(empty)


def test_check_dimension_accepts_compatible_units_and_rejects_mismatch() -> None:
    q = check_dimension("72 km/h", "[velocity]", name="speed")
    assert q.to("m / s").magnitude == pytest.approx(20.0)

    other_ureg = pint.UnitRegistry()
    angle = check_dimension("90 deg", other_ureg.radian, name="angle")
    assert angle.to("rad").magnitude == pytest.approx(1.5707963267948966)

    with pytest.raises(pint.DimensionalityError, match="distance"):
        check_dimension("5 s", "[length]", name="distance")


def test_convert_to_converts_or_raises() -> None:
    converted = convert_to("100 cm", "m", name="length")
    assert converted.magnitude == pytest.approx(1.0)
    assert str(converted.units) == "meter"

    other_ureg = pint.UnitRegistry()
    converted_other = convert_to("36 km/h", other_ureg.meter / other_ureg.second)
    assert converted_other.magnitude == pytest.approx(10.0)

    with pytest.raises(pint.DimensionalityError):
        convert_to("10 kg", "m", name="mass")
