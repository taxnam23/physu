"""Real-world example: firefighter hose targeting and washing machine spin G-force."""

from __future__ import annotations

from physu.constants import STANDARD_GRAVITY
from physu.kinematics import (
    centripetal_acceleration,
    projectile_position,
    projectile_range,
    tangential_speed,
    travel_time,
    velocity_components,
)


def main() -> None:
    """Compute building-wall water stream height and drum spin G-load."""
    # Scenario 1: Aiming a fire hose at a third-floor window 32 m away
    nozzle_speed = "24 m/s"
    aim_angle = "40 deg"
    nozzle_height = "1.2 m"
    building_distance = "32 m"

    vx, _ = velocity_components(nozzle_speed, aim_angle)
    t_reach_wall = travel_time(building_distance, vx)
    x_wall, y_wall = projectile_position(
        nozzle_speed,
        aim_angle,
        t_reach_wall,
        initial_height=nozzle_height,
    )
    max_ground_range = projectile_range(nozzle_speed, aim_angle, initial_height=nozzle_height)

    print("=== Firefighter Hose Targeting ===")
    print(f"Jet reaches wall ({x_wall:.1f}) at t={t_reach_wall:.2f} and height={y_wall:.2f}")
    print(f"Unobstructed ground range: {max_ground_range:.2f}")

    # Scenario 2: Washing machine spin cycle G-force on wet laundry
    drum_radius = "26 cm"
    spin_rate = "1200 rpm"

    rim_speed = tangential_speed(spin_rate, drum_radius)
    a_c = centripetal_acceleration(drum_radius, angular_velocity=spin_rate)
    g_force = (a_c / STANDARD_GRAVITY).to("dimensionless")

    print("\n=== Washing Machine Spin Cycle (1200 rpm) ===")
    print(f"Drum rim speed: {rim_speed:.2f} ({rim_speed.to('km/h'):.1f})")
    print(f"Centripetal acceleration: {a_c:.1f} (~{g_force.magnitude:.0f} g)")


if __name__ == "__main__":
    main()
