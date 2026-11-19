"""Real-world waves & acoustics examples: suspension SHM, guitar, PA dB, ultrasound, and jet."""

from __future__ import annotations

from physu.waves import (
    acoustic_impedance,
    acoustic_reflection_coefficient,
    beat_frequency,
    closed_pipe_harmonic_frequency,
    doppler_frequency,
    fundamental_tension_for_frequency,
    intensity_from_sound_level,
    mach_angle,
    mach_number,
    relativistic_doppler_frequency,
    shm_energy,
    shm_max_speed,
    simple_pendulum_period,
    sound_intensity_level,
    speed_of_sound_ideal_gas,
    spherical_wave_intensity,
    spring_mass_angular_frequency,
    spring_mass_period,
    string_harmonic_frequency,
    string_wave_speed,
)


def main() -> None:
    """Solve practical oscillation, standing wave, acoustic, and Doppler problems."""
    # 1. Car corner suspension SHM + grandfather clock pendulum
    omega_susp = spring_mass_angular_frequency("32 kN / m", "350 kg")
    t_susp = spring_mass_period("350 kg", "32 kN / m")
    e_bump = shm_energy("32 kN / m", "4 cm")
    v_max_susp = shm_max_speed("4 cm", omega_susp)
    t_clock = simple_pendulum_period("99.4 cm")

    print("=== Oscillations: Vehicle Suspension & Clock Pendulum ===")
    print(f"Suspension period: {t_susp:.2f} | Bump energy: {e_bump:.1f} | v_max: {v_max_susp:.2f}")
    print(f"Clock pendulum period (99.4 cm): {t_clock:.3f}")

    # 2. Guitar A4 string (440 Hz), closed organ pipe, and tuning beats
    string_len = "65 cm"
    string_mu = "0.8 g / m"
    tension = fundamental_tension_for_frequency("440 Hz", string_len, string_mu)
    v_string = string_wave_speed(tension, string_mu)
    f3_string = string_harmonic_frequency(3, string_len, v_string)
    c_air = speed_of_sound_ideal_gas(1.4, "20 degC", "28.97 g / mol")
    f1_pipe = closed_pipe_harmonic_frequency(1, "19.5 cm", c_air)
    beats = beat_frequency("440 Hz", "442.5 Hz")

    print("\n=== Standing Waves: Guitar String & Closed Pipe ===")
    print(f"A4 string tension: {tension:.1f} | 3rd harmonic: {f3_string:.0f}")
    print(f"Air sound speed (20 degC): {c_air:.1f} | 19.5 cm closed pipe f1: {f1_pipe:.1f}")
    print(f"Beat frequency (440 Hz vs 442.5 Hz): {beats:.1f}")

    # 3. Outdoor PA loudspeaker dB level & ultrasound air-water reflection
    i_50m = spherical_wave_intensity("25 W", "50 m")
    beta_50m = sound_intensity_level(i_50m)
    i_95db = intensity_from_sound_level("95 dB")
    z_air = acoustic_impedance("1.204 kg / m^3", c_air)
    z_water = acoustic_impedance("1000 kg / m^3", "1480 m / s")
    r_coeff = acoustic_reflection_coefficient(z_air, z_water)

    print("\n=== Acoustics: PA Sound Level & Interface Reflection ===")
    print(f"25 W PA at 50 m: {i_50m.to('mW / m^2'):.3f} ({beta_50m:.1f})")
    print(f"Intensity at 95 dB: {i_95db.to('mW / m^2'):.3f}")
    print(f"Air-water reflection R: {r_coeff:.6f}")

    # 4. Ambulance siren Doppler shift, Mach 1.8 jet, and astronomical redshift
    f_siren_in = doppler_frequency("750 Hz", c_air, source_velocity="110 km / h")
    f_siren_out = doppler_frequency("750 Hz", c_air, source_velocity="-110 km / h")
    m_jet = mach_number("617.4 m / s", c_air)
    mu_jet = mach_angle("617.4 m / s", c_air)
    f_star = relativistic_doppler_frequency("500 THz", "-0.15 c")

    print("\n=== Doppler Effect & Supersonic Flow ===")
    print(f"750 Hz siren @ 110 km/h: approaching {f_siren_in:.1f} | receding {f_siren_out:.1f}")
    print(f"Supersonic jet: Mach {m_jet:.2f} | Mach cone half-angle: {mu_jet.to('deg'):.1f}")
    print(f"Receding star (0.15 c) from 500 THz: {f_star.to('THz'):.1f}")


if __name__ == "__main__":
    main()
