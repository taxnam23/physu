"""Real-world thermodynamics examples: rail gaps, quenching, scuba tank, Carnot, and heat loss."""

from __future__ import annotations

from physu.thermodynamics import (
    carnot_efficiency,
    combined_gas_law,
    conduction_heat_rate,
    equilibrium_temperature,
    ideal_gas_moles,
    linear_thermal_expansion,
    radiation_power,
    sensible_heat,
    temperature_difference,
)


def main() -> None:
    """Solve practical thermal engineering and everyday physics problems."""
    # 1. Steel railway track thermal expansion gap between spring install and summer peak
    rail_length = "25 m"
    steel_alpha = "1.2e-5 / K"
    rail_dt = temperature_difference("15 degC", "48 degC")
    expansion_gap = linear_thermal_expansion(rail_length, steel_alpha, rail_dt)

    print("=== Railway Track Expansion Gap (25 m steel rail) ===")
    print(
        f"Temperature swing: {rail_dt.to('delta_degC'):.1f} | "
        f"Required gap: {expansion_gap.to('mm'):.2f}"
    )

    # 2. Quenching a 2.5 kg hot steel part (450 degC) in 12 kg of water (22 degC)
    water_c = "4184 J / (kg * K)"
    steel_c = "490 J / (kg * K)"
    t_eq = equilibrium_temperature(
        [
            ("2.5 kg", steel_c, "450 degC"),
            ("12 kg", water_c, "22 degC"),
        ]
    )
    water_gain = sensible_heat("12 kg", water_c, temperature_difference("22 degC", t_eq))

    print("\n=== Steel Quenching Bath Equilibrium ===")
    print(f"Final bath temperature: {t_eq.to('degC'):.2f}")
    print(f"Heat absorbed by water: {water_gain.to('kJ'):.1f}")

    # 3. Scuba tank pressure drop entering 8 degC ocean water
    tank_volume = "12 liter"
    fill_pressure = "200 bar"
    fill_temp = "28 degC"
    dive_temp = "8 degC"

    air_moles = ideal_gas_moles(fill_pressure, tank_volume, fill_temp)
    dive_pressure = combined_gas_law(
        fill_pressure,
        tank_volume,
        fill_temp,
        final_volume=tank_volume,
        final_temperature=dive_temp,
    )

    print("\n=== Scuba Cylinder Pressure Check ===")
    print(f"Stored air: {air_moles:.1f} | Pressure at {dive_temp}: {dive_pressure.to('bar'):.1f}")

    # 4. Steam power plant Carnot ceiling & winter window conduction / radiator output
    max_eta = carnot_efficiency("35 degC", "540 degC")
    window_loss = conduction_heat_rate(
        "0.8 W / (m * K)",
        "1.8 m^2",
        temperature_difference("2 degC", "21 degC"),
        "6 mm",
    )
    radiator_net = radiation_power(
        0.92,
        "1.4 m^2",
        "55 degC",
        ambient_temperature="21 degC",
    )

    print("\n=== Power Plant Limit & Home Heating ===")
    print(f"Carnot ceiling (540 degC -> 35 degC): {max_eta.to('%'):.1f}")
    print(f"Single-pane window heat loss: {window_loss.to('kW'):.2f}")
    print(f"Net radiative heat output from radiator (55 degC): {radiator_net:.1f}")


if __name__ == "__main__":
    main()
