"""Unit tests for physu.validation and physu.exceptions."""

from __future__ import annotations

import math

import pint
import pytest

from physu.exceptions import PhysicalDomainError, PhysuError
from physu.units import ureg
from physu.validation import (
    require_non_negative,
    require_non_zero,
    require_positive,
    require_range,
    validate_quantity,
)


def test_exception_hierarchy() -> None:
    assert issubclass(PhysicalDomainError, PhysuError)
    assert issubclass(PhysicalDomainError, ValueError)


def test_validate_quantity_enforces_dimensions_and_domain() -> None:
    q = validate_quantity("500 cm", "[length]", name="length", allow_negative=False)
    assert q.to("m").magnitude == pytest.approx(5.0)

    with pytest.raises(pint.DimensionalityError):
        validate_quantity("5 s", "[length]", name="length")

    with pytest.raises(PhysicalDomainError, match="negative"):
        validate_quantity("-1 s", "[time]", name="time", allow_negative=False)

    with pytest.raises(PhysicalDomainError, match="zero"):
        validate_quantity("0 m/s", "[velocity]", name="velocity", allow_zero=False)

    with pytest.raises(PhysicalDomainError, match="NaN"):
        validate_quantity(float("nan") * ureg.meter, "[length]", name="length")

    with pytest.raises(PhysicalDomainError, match="finite"):
        validate_quantity(float("inf") * ureg.meter, "[length]", name="length")

    inf_q = validate_quantity(
        float("inf") * ureg.meter,
        "[length]",
        name="length",
        allow_infinite=True,
    )
    assert math.isinf(inf_q.magnitude)


def test_scalar_sign_helpers() -> None:
    assert require_non_negative("0 m").magnitude == 0
    assert require_positive("2.5 kg").magnitude == pytest.approx(2.5)
    assert require_non_zero("-3 m/s").magnitude == pytest.approx(-3.0)

    with pytest.raises(PhysicalDomainError):
        require_non_negative("-0.1 m")

    with pytest.raises(PhysicalDomainError):
        require_positive("0 kg")

    with pytest.raises(PhysicalDomainError):
        require_non_zero("0 s")


def test_require_range_with_units_and_exclusivity() -> None:
    q = require_range("50 cm", min_value="0.1 m", max_value="1 m", name="distance")
    assert q.to("m").magnitude == pytest.approx(0.5)

    assert require_range("1 m", min_value="0 m", max_value="1 m").magnitude == 1

    with pytest.raises(PhysicalDomainError):
        require_range(
            "1 m",
            min_value="0 m",
            max_value="1 m",
            inclusive_max=False,
            name="distance",
        )

    with pytest.raises(PhysicalDomainError):
        require_range("-0.1 m", min_value="0 m", name="distance")

    with pytest.raises(pint.DimensionalityError):
        require_range("5 m", min_value="1 s", name="distance")
