"""Real-world electromagnetism examples: feeder wire, flash RC bank, MRI solenoid, and grid."""

from __future__ import annotations

from physu.constants import ELEMENTARY_CHARGE, PROTON_MASS
from physu.electromagnetism import (
    capacitor_energy,
    coulomb_force,
    cyclotron_frequency,
    cyclotron_radius,
    electric_power,
    faraday_emf,
    joule_heating,
    lc_resonant_frequency,
    magnetic_field_solenoid,
    ohms_law_voltage,
    parallel_capacitance,
    rc_charging_voltage,
    rc_time_constant,
    transformer_current,
    transformer_voltage,
    uniform_electric_field,
    voltage_divider,
    wire_resistance,
)


def main() -> None:
    """Solve practical electrical engineering and electromagnetism problems."""
    # 1. Electrostatics & flash capacitor bank (parallel 330 uF + 470 uF charged to 300 V)
    bank_c = parallel_capacitance(["330 uF", "470 uF"])
    flash_energy = capacitor_energy(capacitance=bank_c, voltage="300 V")
    tau_rc = rc_time_constant("2.5 kohm", bank_c)
    v_after_2tau = rc_charging_voltage("300 V", 2.0 * tau_rc, "2.5 kohm", bank_c)
    plate_field = uniform_electric_field("300 V", "15 um")
    dust_force = coulomb_force("2 nC", "-2 nC", "5 mm")

    print("=== Camera Flash Capacitor Bank & Electrostatics ===")
    print(f"Bank capacitance: {bank_c.to('uF'):.0f} | Stored energy: {flash_energy:.1f}")
    print(f"RC time constant: {tau_rc:.2f} | Voltage at 2*tau: {v_after_2tau:.1f}")
    print(f"Dielectric field (15 um gap): {plate_field.to('MV / m'):.1f}")
    print(f"Static attraction (2 nC, -2 nC @ 5 mm): {dust_force.to('mN'):.2f}")

    # 2. Workshop 24 V copper feeder cable + thermistor voltage divider
    cable_r = wire_resistance("1.68e-8 ohm * m", "40 m", "2.5 mm^2")
    load_current = "8 A"
    voltage_drop = ohms_law_voltage(load_current, cable_r)
    line_loss = electric_power(current=load_current, resistance=cable_r)
    cable_heat_1h = joule_heating(load_current, cable_r, "1 hour")
    adc_v = voltage_divider("5 V", "10 kohm", "15 kohm")

    print("\n=== DC Feeder Cable & Sensor Divider ===")
    print(f"40 m copper loop R: {cable_r:.3f} | Drop at 8 A: {voltage_drop:.2f}")
    print(f"Line loss: {line_loss:.1f} | 1 h thermal loss: {cable_heat_1h.to('Wh'):.1f}")
    print(f"Voltage divider output (5 V, 10k/15k): {adc_v:.2f}")

    # 3. Proton therapy cyclotron & superconducting solenoid
    sol_b = magnetic_field_solenoid(1200, "0.8 m", "800 A")
    proton_r = cyclotron_radius(PROTON_MASS, "1.5e7 m/s", ELEMENTARY_CHARGE, sol_b)
    proton_f = cyclotron_frequency(ELEMENTARY_CHARGE, sol_b, PROTON_MASS)

    print("\n=== Solenoid & Cyclotron Beam ===")
    print(f"Solenoid B: {sol_b:.2f}")
    print(f"Proton orbit radius: {proton_r.to('cm'):.2f} | f_c: {proton_f.to('MHz'):.2f}")

    # 4. Step-down transformer, induction coil, and RF LC resonant tank
    sec_v = transformer_voltage("13.8 kV", 2300, 40)
    sec_i = transformer_current("12 A", 2300, 40)
    coil_emf = faraday_emf("-8 mWb", "5 ms", turns=250)
    tank_f0 = lc_resonant_frequency("15 uH", "68 pF")

    print("\n=== Transformer, Induction & LC Tank ===")
    print(f"Transformer output: {sec_v:.1f} at {sec_i:.1f}")
    print(f"Induced EMF (250 turns, -8 mWb in 5 ms): {coil_emf:.0f}")
    print(f"RF tank resonance (15 uH, 68 pF): {tank_f0.to('MHz'):.2f}")


if __name__ == "__main__":
    main()
