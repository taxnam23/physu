"""Real-world example: highway emergency braking check and bridge clearance drop."""

from __future__ import annotations

from physu.kinematics import (
    accelerated_displacement,
    displacement,
    free_fall_speed,
    free_fall_time,
    time_to_velocity,
)


def main() -> None:
    """Analyze safe highway braking distance and a dropped inspection sensor."""
    # Scenario 1: Emergency braking on a wet highway
    cruise_speed = "110 km/h"
    reaction_time = "1.2 s"
    braking_decel = "-6.5 m/s^2"
    hazard_distance = "95 m"

    reaction_dist = displacement(cruise_speed, reaction_time)
    braking_time = time_to_velocity(cruise_speed, "0 m/s", braking_decel)
    braking_dist = accelerated_displacement(cruise_speed, braking_decel, braking_time)
    total_stopping_dist = reaction_dist + braking_dist

    print("=== Highway Emergency Braking ===")
    print(f"Cruise speed: {cruise_speed} | Reaction distance (1.2 s): {reaction_dist:.2f}")
    print(f"Braking time: {braking_time:.2f} | Braking distance: {braking_dist:.2f}")
    print(f"Total stopping distance: {total_stopping_dist:.2f} (hazard at {hazard_distance})")
    margin = total_stopping_dist.to("m").magnitude
    print(f"Clears hazard before impact? {'YES' if margin < 95.0 else 'NO'}")

    # Scenario 2: Dropping an acoustic depth probe from a 38 m bridge
    bridge_height = "38 m"
    t_splash = free_fall_time(bridge_height)
    v_splash = free_fall_speed(bridge_height)

    print("\n=== Bridge Drop Sensor Check ===")
    print(f"Drop height: {bridge_height}")
    print(f"Time to water impact: {t_splash:.2f}")
    print(f"Water entry speed: {v_splash:.2f} ({v_splash.to('km/h'):.1f})")


if __name__ == "__main__":
    main()
