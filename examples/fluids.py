"""Real-world fluid mechanics examples: diving, iceberg, Venturi, Poiseuille, and capillarity."""

from __future__ import annotations

from physu.fluids import (
    apparent_weight,
    bernoulli_pressure,
    capillary_rise_height,
    continuity_speed,
    floating_submerged_fraction,
    hydrostatic_pressure,
    kinematic_viscosity,
    laplace_bubble_pressure,
    pascal_hydraulic_force,
    poiseuille_flow_rate,
    reynolds_number,
    stokes_terminal_velocity,
    torricelli_efflux_speed,
    volumetric_flow_rate,
)


def main() -> None:
    """Solve practical hydrostatics, buoyancy, flow, viscosity, and capillarity problems."""
    # 1. Deep-sea submersible pressure, hydraulic lift, and floating iceberg
    seawater_rho = "1025 kg / m^3"
    p_300m = hydrostatic_pressure(seawater_rho, "300 m")
    f_lift = pascal_hydraulic_force("350 N", "10 cm^2", "450 cm^2")
    iceberg_sub = floating_submerged_fraction("917 kg / m^3", seawater_rho)
    anchor_w_app = apparent_weight("500 kg", "64 liter", seawater_rho)

    print("=== Hydrostatics & Buoyancy ===")
    print(f"Pressure at 300 m ocean depth: {p_300m.to('bar'):.1f}")
    print(f"Hydraulic lift output (350 N, 10->450 cm^2): {f_lift.to('kN'):.2f}")
    print(f"Iceberg submerged fraction: {iceberg_sub.to('%'):.1f}")
    print(f"500 kg steel anchor apparent weight underwater: {anchor_w_app:.0f}")

    # 2. Venturi water meter and reservoir orifice efflux
    q_pipe = volumetric_flow_rate("25 cm^2", "2.0 m / s")
    v_throat = continuity_speed("25 cm^2", "2.0 m / s", "8 cm^2")
    p_throat = bernoulli_pressure("250 kPa", "1000 kg / m^3", "2.0 m / s", "0 m", v_throat, "0 m")
    v_tank = torricelli_efflux_speed("12 m")

    print("\n=== Fluid Dynamics: Venturi Meter & Torricelli ===")
    print(f"Pipe flow: {q_pipe.to('L / min'):.0f} | Throat speed: {v_throat:.2f}")
    print(f"Venturi throat pressure (from 250 kPa): {p_throat.to('kPa'):.1f}")
    print(f"Tank efflux speed at 12 m depth: {v_tank:.2f}")

    # 3. SAE 30 oil viscosity, Reynolds number, Poiseuille capillary, and Stokes settling
    nu_oil = kinematic_viscosity("290 cP", "890 kg / m^3")
    re_oil = reynolds_number("890 kg / m^3", "0.8 m / s", "2.5 cm", "290 cP")
    q_artery = poiseuille_flow_rate("1.5 kPa", "1.5 mm", "3.5 cP", "10 cm")
    v_sediment = stokes_terminal_velocity("0.2 mm", "2650 kg / m^3", "1000 kg / m^3", "1.0 cP")

    print("\n=== Viscosity, Reynolds & Stokes ===")
    print(f"SAE 30 kinematic viscosity: {nu_oil.to('cSt'):.1f} | Re = {re_oil:.1f}")
    print(f"Laminar tube flow (1.5 mm radius, 1.5 kPa): {q_artery.to('mL / min'):.1f}")
    print(f"Sand grain (0.2 mm) Stokes settling in water: {v_sediment.to('cm / s'):.2f}")

    # 4. Soap bubble excess pressure and glass capillary rise/depression
    dp_bubble = laplace_bubble_pressure("25 mN / m", "2 cm", interfaces=2)
    h_water = capillary_rise_height("72.8 mN / m", "1000 kg / m^3", "0.4 mm")
    h_mercury = capillary_rise_height(
        "486.5 mN / m",
        "13.5951 g / cm^3",
        "0.4 mm",
        contact_angle="140 deg",
    )

    print("\n=== Capillarity: Soap Bubble & Meniscus ===")
    print(f"2 cm soap bubble excess pressure: {dp_bubble:.2f}")
    print(f"0.4 mm tube rise: water {h_water.to('cm'):.2f} | mercury {h_mercury.to('cm'):.2f}")


if __name__ == "__main__":
    main()
