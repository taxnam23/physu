"""Real-world example: loading a heavy crate up an inclined steel ramp."""

from __future__ import annotations

from physu.dynamics import (
    critical_incline_angle,
    inclined_normal_force,
    inclined_parallel_force,
    kinetic_friction,
    max_static_friction,
)


def main() -> None:
    """Evaluate slip risk and winch pulling forces on a 18-degree loading ramp."""
    crate_mass = "450 kg"
    ramp_angle = "18 deg"
    mu_static = 0.42
    mu_kinetic = 0.30

    repose_angle = critical_incline_angle(mu_static).to("deg")
    gravity_pull = inclined_parallel_force(crate_mass, ramp_angle)
    normal = inclined_normal_force(crate_mass, ramp_angle)
    max_hold = max_static_friction(mu_static, normal)
    sliding_friction = kinetic_friction(mu_kinetic, normal)

    # Pulling force up the ramp must overcome gravity component + friction
    breakaway_pull = gravity_pull + max_hold
    steady_winch_pull = gravity_pull + sliding_friction

    print("=== Inclined Ramp Winch Sizing (450 kg crate, 18 deg) ===")
    print(f"Self-slip threshold (angle of repose): {repose_angle:.2f}")
    holds_stationary = "YES" if repose_angle.magnitude > 18.0 else "NO"
    print(f"Crate holds stationary without brakes? {holds_stationary}")
    print(f"Gravity component down ramp: {gravity_pull:.1f} | Normal force: {normal:.1f}")
    print(f"Breakaway winch force needed: {breakaway_pull:.1f}")
    print(f"Steady winch pull while moving: {steady_winch_pull:.1f}")


if __name__ == "__main__":
    main()
