"""Examples covering Newton's laws, inclined planes, and static/kinetic friction."""

from __future__ import annotations

from physu.dynamics import (
    acceleration,
    critical_incline_angle,
    force,
    friction_coefficient,
    inclined_normal_force,
    inclined_parallel_force,
    kinetic_friction,
    mass,
    max_static_friction,
    normal_force,
    weight,
)


def main() -> None:
    """Run Newton's laws and friction examples."""
    # 1. Newton's Laws & Inclined Planes
    f_net = force("2000 g", "3 m/s^2")
    acc = acceleration("15 N", "3 kg")
    m_obj = mass("20 N", "4 m/s^2")
    w = weight("10 kg")
    f_par = inclined_parallel_force("10 kg", "30 deg")
    f_norm = inclined_normal_force("10 kg", "30 deg")
    print(f"[Newton] F={f_net}, a={acc}, m={m_obj}, W={w:.2f}")
    print(f"[Incline 30 deg] F_parallel={f_par:.2f}, N={f_norm:.2f}")

    # 2. Static & Kinetic Friction
    n_flat = normal_force("10 kg")
    n_incline = normal_force("10 kg", angle="30 deg")
    fs_max = max_static_friction(0.5, n_incline)
    fk = kinetic_friction(0.3, n_incline)
    mu = friction_coefficient("30 N", "150 N")
    theta_c = critical_incline_angle(0.5)
    print(f"[Friction] N_flat={n_flat:.2f}, N_incline={n_incline:.2f}")
    print(
        f"[Friction] fs_max={fs_max:.2f}, fk={fk:.2f}, "
        f"mu={mu:.2f}, critical_angle={theta_c.to('deg'):.2f}"
    )


if __name__ == "__main__":
    main()
