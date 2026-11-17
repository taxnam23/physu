"""Real-world examples: crane hoist motor sizing, spring coaster launch, and escape speed."""

from __future__ import annotations

from physu.energy import (
    average_power,
    elastic_potential_energy,
    escape_speed,
    final_speed_conservation,
    gravitational_potential_energy,
    speed_from_kinetic_energy,
)
from physu.units import to_unit


def main() -> None:
    """Solve practical engineering energy and power sizing problems."""
    # 1. Construction crane lifting a 1.2-tonne steel I-beam up 18 m in 25 s
    beam_mass = "1200 kg"
    lift_height = "18 m"
    lift_duration = "25 s"
    hoist_efficiency = 0.82

    useful_energy = gravitational_potential_energy(beam_mass, lift_height)
    useful_power = average_power(useful_energy, lift_duration)
    required_motor_power = to_unit(useful_power / hoist_efficiency, "kW")
    electrical_energy_per_lift = to_unit(useful_energy / hoist_efficiency, "kWh")

    print("=== Construction Crane Hoist Sizing ===")
    print(f"Useful lifting energy: {useful_energy.to('kJ'):.1f}")
    print(f"Output shaft power: {useful_power.to('kW'):.2f}")
    print(
        f"Required motor input (@82% eff.): {required_motor_power:.2f} "
        f"({required_motor_power.to('hp'):.2f})"
    )
    print(f"Grid energy per lift: {electrical_energy_per_lift:.4f}")

    # 2. Spring-launched coaster cart climbing a 12 m hill
    cart_mass = "250 kg"
    launcher_k = "32 kN/m"
    spring_compression = "1.6 m"
    hill_height = "12 m"

    stored_elastic = elastic_potential_energy(launcher_k, spring_compression)
    launch_speed = speed_from_kinetic_energy(stored_elastic, cart_mass)
    crest_speed = final_speed_conservation(launch_speed, "0 m", hill_height)

    print("\n=== Spring-Launched Coaster Cart ===")
    print(f"Stored launcher energy: {stored_elastic.to('kJ'):.2f}")
    print(f"Launch speed: {launch_speed:.2f} ({launch_speed.to('km/h'):.1f})")
    print(f"Speed clearing 12 m crest (frictionless): {crest_speed:.2f}")

    # 3. Minimum Earth escape speed from a 500 km orbital parking altitude
    v_esc = escape_speed("5.9722e24 kg", "6871 km")
    print("\n=== Deep-Space Departure from 500 km LEO ===")
    print(f"Escape speed needed: {v_esc.to('km/s'):.2f}")


if __name__ == "__main__":
    main()
