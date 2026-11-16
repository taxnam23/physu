"""Examples covering 2D projectile motion and uniform circular motion (UCM)."""

from __future__ import annotations

from physu.constants import STANDARD_GRAVITY
from physu.kinematics import (
    angular_displacement,
    angular_velocity_from_frequency,
    arc_length,
    centripetal_acceleration,
    frequency_from_period,
    period_from_angular_velocity,
    projectile_flight_time,
    projectile_max_height,
    projectile_position,
    projectile_range,
    projectile_velocity,
    tangential_speed,
    velocity_components,
)


def main() -> None:
    """Run projectile motion and uniform circular motion examples."""
    # 1. 2D Projectile Motion
    vx0, vy0 = velocity_components("20 m/s", "45 deg")
    t_flight = projectile_flight_time("20 m/s", "45 deg", gravity=STANDARD_GRAVITY)
    h_max = projectile_max_height("20 m/s", "45 deg", gravity=STANDARD_GRAVITY)
    rng = projectile_range("20 m/s", "45 deg", gravity=STANDARD_GRAVITY)
    pos_1s = projectile_position("20 m/s", "45 deg", "1 s")
    vel_1s = projectile_velocity("20 m/s", "45 deg", "1 s")
    print(f"[Projectile] vx0={vx0:.2f}, vy0={vy0:.2f}")
    print(f"[Projectile] flight_time={t_flight:.2f}, max_height={h_max:.2f}, range={rng:.2f}")
    print(
        f"[Projectile] at t=1s: pos=({pos_1s[0]:.2f}, {pos_1s[1]:.2f}), "
        f"vel=({vel_1s[0]:.2f}, {vel_1s[1]:.2f})"
    )

    # 2. Uniform Circular Motion (UCM)
    omega = angular_velocity_from_frequency("2 Hz")
    period = period_from_angular_velocity("60 rpm")
    freq = frequency_from_period(period)
    v_tan = tangential_speed("60 rpm", "2 m")
    a_c = centripetal_acceleration("2 m", angular_velocity="60 rpm")
    d_theta = angular_displacement("60 rpm", "0.5 s")
    s_arc = arc_length("2 m", "90 deg")
    print(f"[Circular] omega(2Hz)={omega:.2f}, T(60rpm)={period:.2f}, f={freq:.2f}")
    print(f"[Circular] v_tan={v_tan:.2f}, a_c={a_c:.2f}, d_theta={d_theta:.2f}, arc={s_arc:.2f}")


if __name__ == "__main__":
    main()
