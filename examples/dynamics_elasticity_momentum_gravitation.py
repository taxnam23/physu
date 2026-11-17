"""Real-world examples: vehicle suspension sag, rear-end impact, and CubeSat orbit."""

from __future__ import annotations

from physu.dynamics import (
    average_force,
    centripetal_force,
    gravitational_field_strength,
    inelastic_collision_velocity,
    momentum_change,
    newton,
    orbital_speed,
    parallel_spring_constant,
    spring_displacement,
)


def main() -> None:
    """Run three practical dynamics calculations."""
    # 1. Rear suspension sag on a loaded utility van (two parallel coil springs)
    rear_axle_load = newton.weight("650 kg")
    k_eq = parallel_spring_constant(["28 kN/m", "28 kN/m"])
    # Upward spring restoring force balances downward axle weight (+rear_axle_load)
    sag = spring_displacement(rear_axle_load, k_eq)

    print("=== Rear Suspension Compression ===")
    print(f"Combined stiffness: {k_eq.to('kN/m'):.1f} | Sag under 650 kg: {sag.to('cm'):.2f}")

    # 2. Low-speed rear-end collision force estimate (inelastic bumper lock)
    car1_mass = "1500 kg"
    car1_speed = "36 km/h"
    car2_mass = "1200 kg"
    impact_duration = "120 ms"

    v_after = inelastic_collision_velocity(car1_mass, car1_speed, car2_mass, "0 m/s")
    impulse_car2 = momentum_change(car2_mass, "0 m/s", v_after)
    avg_impact_force = average_force(impulse_car2, impact_duration)

    print("\n=== Low-Speed Rear-End Collision ===")
    print(f"Shared speed right after impact: {v_after.to('km/h'):.1f}")
    print(f"Average force over {impact_duration}: {avg_impact_force.to('kN'):.2f}")

    # 3. 12 kg CubeSat in a 550 km Low Earth Orbit (Earth radius 6371 km)
    earth_mass = "5.9722e24 kg"
    orbit_radius = "6921 km"
    cubesat_mass = "12 kg"

    local_g = gravitational_field_strength(earth_mass, orbit_radius)
    v_orbit = orbital_speed(earth_mass, orbit_radius)
    f_gravity_orbit = centripetal_force(cubesat_mass, orbit_radius, tangential_speed=v_orbit)

    print("\n=== CubeSat Orbit Check (550 km altitude) ===")
    print(f"Local gravity at 550 km: {local_g:.2f}")
    print(f"Orbital velocity: {v_orbit.to('km/s'):.2f}")
    print(f"Centripetal gravitational force on 12 kg CubeSat: {f_gravity_orbit:.2f}")


if __name__ == "__main__":
    main()
