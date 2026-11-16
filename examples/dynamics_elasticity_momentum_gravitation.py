"""Examples covering elasticity (Hooke's law), linear momentum/collisions, and gravitation."""

from __future__ import annotations

from physu.dynamics import (
    average_force,
    centripetal_force,
    collision_velocities,
    elastic_collision_velocities,
    gravitational_field_strength,
    gravitational_force,
    impulse,
    inelastic_collision_velocity,
    linear_momentum,
    momentum_change,
    orbital_speed,
    parallel_spring_constant,
    series_spring_constant,
    spring_constant,
    spring_displacement,
    spring_force,
    spring_force_magnitude,
)


def main() -> None:
    """Run elasticity, momentum, and gravitation/circular dynamics examples."""
    # 1. Elasticity & Springs (Hooke's Law)
    f_restoring = spring_force("200 N/m", "15 cm")
    f_mag = spring_force_magnitude("200 N/m", "-15 cm")
    k_calc = spring_constant("-30 N", "15 cm")
    dx = spring_displacement("-30 N", "200 N/m")
    k_ser = series_spring_constant(["100 N/m", "200 N/m"])
    k_par = parallel_spring_constant(["100 N/m", "200 N/m"])
    print(f"[Elasticity] F={f_restoring}, |F|={f_mag}, k={k_calc}, x={dx}")
    print(f"[Elasticity] k_series={k_ser:.2f}, k_parallel={k_par:.2f}")

    # 2. Linear Momentum, Impulse & 1D Collisions
    p = linear_momentum("2 kg", "72 km/h")
    j = impulse("50 N", "0.4 s")
    dp = momentum_change("2 kg", "-3 m/s", "5 m/s")
    f_avg = average_force(j, "0.4 s")
    v1_el, v2_el = elastic_collision_velocities("1 kg", "4 m/s", "3 kg", "0 m/s")
    v_inel = inelastic_collision_velocity("2 kg", "6 m/s", "4 kg", "-3 m/s")
    v1_p, v2_p = collision_velocities("2 kg", "6 m/s", "2 kg", "0 m/s", restitution=0.5)
    print(f"[Momentum] p={p}, J={j}, dp={dp}, F_avg={f_avg}")
    print(f"[Collisions] elastic=({v1_el}, {v2_el}), inelastic={v_inel}, e=0.5=({v1_p}, {v2_p})")

    # 3. Universal Gravitation & Circular Dynamics
    g_earth = gravitational_field_strength("5.9722e24 kg", "6371 km")
    f_grav = gravitational_force("5.9722e24 kg", "70 kg", "6371 km")
    v_leo = orbital_speed("5.9722e24 kg", "6771 km")
    f_cent = centripetal_force("1000 kg", "50 m", tangential_speed="72 km/h")
    print(f"[Gravitation] g_earth={g_earth:.2f}, F_grav(70kg)={f_grav:.2f}")
    print(f"[Circular Dynamics] v_LEO={v_leo.to('km/s'):.2f}, F_c={f_cent:.2f}")


if __name__ == "__main__":
    main()
