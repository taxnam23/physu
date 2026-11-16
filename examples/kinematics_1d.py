"""Examples covering 1D kinematics: uniform, accelerated (MRUA/SUVAT), and vertical motion."""

from __future__ import annotations

from physu import PhysicalDomainError, ureg
from physu.kinematics import (
    accelerated_displacement,
    acceleration,
    average_velocity,
    displacement,
    final_velocity,
    free_fall_speed,
    free_fall_time,
    max_height,
    position,
    time_to_height,
    time_to_peak,
    torricelli_velocity,
    travel_time,
)


def main() -> None:
    """Run 1D uniform, accelerated, and vertical motion examples."""
    # 1. Uniform Motion (MRU)
    dx = displacement("72 km/h", "10 s")
    pos = position("10 m", "5 m/s", "4 s")
    v_avg = average_velocity("150 m", "6 s")
    t_trip = travel_time("300 km", "100 km/h")
    print(f"[Uniform] dx={dx}, pos={pos}, v_avg={v_avg}, t_trip={t_trip.to('hour')}")

    # 2. Uniformly Accelerated Motion (MRUA / SUVAT)
    v_f = final_velocity(10 * (ureg.m / ureg.s), "2 m/s^2", "5 s")
    dx_acc = accelerated_displacement("10 m/s", "2 m/s^2", "5 s")
    acc = acceleration("0 m/s", "25 m/s", "5 s")
    v_torr = torricelli_velocity("0 m/s", "2 m/s^2", "25 m")
    print(f"[Accelerated] v_f={v_f}, dx={dx_acc}, a={acc}, |v_torr|={v_torr}")

    # 3. Free Fall & Vertical Throw
    t_fall = free_fall_time("45 m", gravity="10 m/s^2")
    v_impact = free_fall_speed("45 m", gravity="10 m/s^2")
    t_peak = time_to_peak("20 m/s", gravity="10 m/s^2")
    h_max = max_height("20 m/s", initial_height="5 m", gravity="10 m/s^2")
    up_down_times = time_to_height("20 m/s", "15 m", gravity="10 m/s^2")
    print(
        f"[Vertical] t_fall={t_fall}, v_impact={v_impact}, "
        f"t_peak={t_peak}, h_max={h_max}, times@15m={up_down_times}"
    )

    try:
        time_to_height("20 m/s", "30 m", gravity="10 m/s^2")
    except PhysicalDomainError as exc:
        print("[Vertical] Unreachable height caught:", exc)


if __name__ == "__main__":
    main()
