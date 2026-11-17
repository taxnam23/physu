"""Real-world example: validating and normalizing mixed-unit drone telemetry payloads."""

from __future__ import annotations

import pint

from physu import PhysicalDomainError, Quantity, ureg
from physu.constants import STANDARD_GRAVITY
from physu.units import to_si, to_unit
from physu.validation import require_range, validate_quantity


def parse_telemetry_packet(
    payload: dict[str, object],
) -> tuple[pint.Quantity, pint.Quantity, pint.Quantity]:
    """Validate incoming drone telemetry and normalize units for the flight controller."""
    battery_mass = validate_quantity(
        payload["battery_mass"],  # type: ignore[arg-type]
        "[mass]",
        name="battery_mass",
        allow_negative=False,
        allow_zero=False,
    )
    airspeed = validate_quantity(
        payload["airspeed"],  # type: ignore[arg-type]
        "[velocity]",
        name="airspeed",
        allow_negative=False,
    )
    pitch_angle = require_range(
        validate_quantity(payload["pitch"], "rad", name="pitch"),  # type: ignore[arg-type]
        min_value=Quantity(-45, "deg"),
        max_value=Quantity(45, "deg"),
        name="pitch",
    )
    return to_unit(battery_mass, "kg"), to_si(airspeed), to_unit(pitch_angle, "deg")


def main() -> None:
    """Validate valid and faulty telemetry packets from heterogeneous sensors."""
    external_sensor_reg = pint.UnitRegistry()

    valid_packet: dict[str, object] = {
        "battery_mass": external_sensor_reg.Quantity(1450, "g"),
        "airspeed": "65 km/h",
        "pitch": 12 * ureg.deg,
    }
    mass, speed, pitch = parse_telemetry_packet(valid_packet)
    hover_weight = to_unit(mass * STANDARD_GRAVITY, "N")
    print(f"Valid telemetry -> mass={mass}, airspeed={speed:.2f}, pitch={pitch}")
    print(f"Battery weight load on frame: {hover_weight:.2f}")

    corrupted_packets: list[tuple[str, dict[str, object]]] = [
        (
            "unit mismatch (pressure sent instead of speed)",
            {"battery_mass": "1.5 kg", "airspeed": "101 kPa", "pitch": "10 deg"},
        ),
        (
            "out-of-bounds pitch (nose dive)",
            {"battery_mass": "1.5 kg", "airspeed": "15 m/s", "pitch": "-70 deg"},
        ),
    ]
    for label, packet in corrupted_packets:
        try:
            parse_telemetry_packet(packet)
        except (pint.DimensionalityError, PhysicalDomainError) as exc:
            print(f"Rejected {label}: {exc}")


if __name__ == "__main__":
    main()
