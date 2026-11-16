"""Examples covering physu.units, physu.constants, physu.validation, and exceptions."""

from __future__ import annotations

import pint

from physu import PhysicalDomainError, Quantity, ureg
from physu.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, STANDARD_GRAVITY
from physu.units import convert_to, to_quantity, to_si, to_unit
from physu.validation import require_range, validate_quantity


def main() -> None:
    """Demonstrate unit normalization, constants, and dimensional/domain validation."""
    # Units and conversion helpers
    speed = to_quantity("72 km/h")
    print("Speed:", speed)
    print("Speed in m/s:", to_si(speed))
    print("Speed in mph:", round(convert_to(speed, "mph"), 2))

    external_reg = pint.UnitRegistry()
    normalized = to_unit(external_reg.Quantity(1500, "g"), "kg")
    print("Normalized from external registry:", normalized)

    # Built-in physical constants
    print("Standard gravity:", STANDARD_GRAVITY)
    print("Speed of light:", SPEED_OF_LIGHT)
    print("Gravitational constant:", GRAVITATIONAL_CONSTANT)

    # Validation helpers
    mass = validate_quantity(5 * ureg.kg, "[mass]", name="mass", allow_negative=False)
    incline = require_range(
        Quantity(30, "deg"),
        min_value=Quantity(0, "rad"),
        max_value=Quantity(90, "deg"),
        name="incline",
    )
    print("Validated mass & angle:", mass, incline.to("rad"))

    # Catching dimensional vs domain errors
    try:
        validate_quantity("10 kg", "[velocity]", name="speed")
    except pint.DimensionalityError as exc:
        print("Dimensional error caught:", exc)

    try:
        validate_quantity("-5 s", "[time]", name="time", allow_negative=False)
    except PhysicalDomainError as exc:
        print("Physical domain error caught:", exc)


if __name__ == "__main__":
    main()
