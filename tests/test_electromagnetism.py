"""Unit tests for physu.electromagnetism submodules."""

from __future__ import annotations

import math

import pint
import pytest

from physu import PhysicalDomainError, electromagnetism
from physu.constants import (
    COULOMB_CONSTANT,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    VACUUM_PERMEABILITY,
    VACUUM_PERMITTIVITY,
)
from physu.electromagnetism import (
    capacitance,
    capacitor_charge,
    capacitor_energy,
    capacitor_voltage,
    coulomb_force,
    cyclotron_frequency,
    cyclotron_radius,
    electric_field_point_charge,
    electric_flux,
    electric_force,
    electric_potential_energy,
    electric_potential_point_charge,
    electric_power,
    electric_work,
    faraday_emf,
    force_per_length_parallel_wires,
    gauss_electric_flux,
    inductor_energy,
    joule_heating,
    lc_resonant_frequency,
    lorentz_magnetic_force,
    magnetic_field_loop_center,
    magnetic_field_solenoid,
    magnetic_field_wire,
    magnetic_flux,
    magnetic_force_on_wire,
    motional_emf,
    ohms_law_current,
    ohms_law_resistance,
    ohms_law_voltage,
    parallel_capacitance,
    parallel_inductance,
    parallel_plate_capacitance,
    parallel_resistance,
    rc_charging_voltage,
    rc_discharging_voltage,
    rc_time_constant,
    rl_time_constant,
    series_capacitance,
    series_inductance,
    series_resistance,
    solenoid_inductance,
    transformer_current,
    transformer_voltage,
    uniform_electric_field,
    voltage_divider,
    wire_resistance,
)


def test_coulomb_and_electric_field_calculations() -> None:
    k_e = float(COULOMB_CONSTANT.to("N * m^2 / C^2").magnitude)

    # Like charges: +2 uC and +3 uC at 10 cm in vacuum -> repulsive force
    f_rep = coulomb_force("2 uC", "3 uC", "10 cm")
    assert str(f_rep.units) == "newton"
    assert f_rep.magnitude == pytest.approx(k_e * 2e-6 * 3e-6 / 0.1**2)

    # Opposite charges in dielectric (eps_r = 4) -> attractive and scaled down
    f_attr = coulomb_force("2 uC", "-3 uC", "10 cm", relative_permittivity=4.0)
    assert f_attr.magnitude == pytest.approx(-f_rep.magnitude / 4.0)

    # Zero charge yields clean 0.0 N
    f_zero = coulomb_force("0 C", "-3 uC", "10 cm")
    assert f_zero.magnitude == 0.0

    # Point charge electric field (V / m == N / C) and electric force F = q * E
    e_field = electric_field_point_charge("-5 nC", "20 cm")
    assert str(e_field.units) == "volt / meter"
    assert e_field.to("N / C").magnitude == pytest.approx(-k_e * 5e-9 / 0.2**2)

    f_on_test = electric_force("4 uC", "2.5 kN / C")
    assert f_on_test.to("mN").magnitude == pytest.approx(10.0)


def test_uniform_field_potential_energy_and_work() -> None:
    k_e = float(COULOMB_CONSTANT.to("N * m^2 / C^2").magnitude)

    e_plates = uniform_electric_field("1.5 kV", "5 mm")
    assert str(e_plates.units) == "volt / meter"
    assert e_plates.to("kV / m").magnitude == pytest.approx(300.0)

    v_point = electric_potential_point_charge("4 uC", "2 m")
    assert str(v_point.units) == "volt"
    assert v_point.magnitude == pytest.approx(k_e * 4e-6 / 2.0)

    u_pair = electric_potential_energy("2 uC", "-3 uC", "0.5 m", relative_permittivity=2.0)
    assert str(u_pair.units) == "joule"
    assert u_pair.magnitude == pytest.approx(k_e * 2e-6 * (-3e-6) / (2.0 * 0.5))

    # Electron accelerated across +1 kV potential increase:
    # W_e = -(-e)(+1000 V) = +1 keV
    w_elec = electric_work(-ELEMENTARY_CHARGE, "1 kV")
    assert str(w_elec.units) == "joule"
    assert w_elec.to("keV").magnitude == pytest.approx(1.0)
    assert electric_work("0 C", "100 V").magnitude == 0.0


def test_electric_flux_and_gauss_law() -> None:
    flux_perp = electric_flux("500 N / C", "200 cm^2")
    assert str(flux_perp.units) == "meter * volt"
    assert flux_perp.magnitude == pytest.approx(10.0)

    flux_angled = electric_flux("500 V / m", "2 m^2", angle="60 deg")
    assert flux_angled.magnitude == pytest.approx(500.0)

    flux_zero = electric_flux("500 V / m", "2 m^2", angle="90 deg")
    assert flux_zero.magnitude == 0.0

    q_enc = 1.77083756376e-9
    gauss_flux = gauss_electric_flux(f"{q_enc} C")
    eps_0 = float(VACUUM_PERMITTIVITY.to("F / m").magnitude)
    assert str(gauss_flux.units) == "meter * volt"
    assert gauss_flux.magnitude == pytest.approx(q_enc / eps_0)

    # Sphere Gauss check: E(r) * 4*pi*r^2 == Q_enc / (eps_0 * eps_r)
    q = "3 uC"
    r = 0.4
    eps_r = 2.5
    e_r = electric_field_point_charge(q, f"{r} m", relative_permittivity=eps_r)
    sphere_flux = electric_flux(e_r, f"{4.0 * math.pi * r**2} m^2")
    assert sphere_flux.magnitude == pytest.approx(
        gauss_electric_flux(q, relative_permittivity=eps_r).magnitude
    )


def test_electrostatics_validation_errors() -> None:
    # Dimensionality mismatches
    with pytest.raises(pint.DimensionalityError):
        coulomb_force("2 A", "3 C", "1 m")
    with pytest.raises(pint.DimensionalityError):
        electric_field_point_charge("2 C", "1 s")
    with pytest.raises(pint.DimensionalityError):
        electric_force("1 C", "10 T")
    with pytest.raises(pint.DimensionalityError):
        uniform_electric_field("12 A", "1 cm")
    with pytest.raises(pint.DimensionalityError):
        electric_potential_point_charge("1 C", "2 m", relative_permittivity="2 m")
    with pytest.raises(pint.DimensionalityError):
        electric_work("1 C", "10 N")
    with pytest.raises(pint.DimensionalityError):
        electric_flux("100 V / m", "2 m")
    with pytest.raises(pint.DimensionalityError):
        electric_flux("100 V / m", "2 m^2", angle="10 m")
    with pytest.raises(pint.DimensionalityError):
        gauss_electric_flux("5 V")

    # Physical domain violations
    with pytest.raises(PhysicalDomainError):
        coulomb_force("1 C", "1 C", "0 m")
    with pytest.raises(PhysicalDomainError):
        coulomb_force("1 C", "1 C", "-0.5 m")
    with pytest.raises(PhysicalDomainError):
        electric_field_point_charge("1 C", "1 m", relative_permittivity=0)
    with pytest.raises(PhysicalDomainError):
        electric_field_point_charge("1 C", "1 m", relative_permittivity=-2)
    with pytest.raises(PhysicalDomainError):
        uniform_electric_field("12 V", "0 m")
    with pytest.raises(PhysicalDomainError):
        electric_potential_point_charge("1 C", "-1 cm")
    with pytest.raises(PhysicalDomainError):
        electric_flux("100 V / m", "-2 m^2")
    with pytest.raises(PhysicalDomainError):
        gauss_electric_flux("1 uC", relative_permittivity=0)


def test_ohms_law_wire_power_and_joule_heating() -> None:
    v = ohms_law_voltage("250 mA", "4.7 kohm")
    assert str(v.units) == "volt"
    assert v.magnitude == pytest.approx(1175.0)

    i = ohms_law_current("12 V", "4 ohm")
    assert str(i.units) == "ampere"
    assert i.magnitude == pytest.approx(3.0)

    r = ohms_law_resistance("24 V", "600 mA")
    assert str(r.units) == "ohm"
    assert r.magnitude == pytest.approx(40.0)
    assert ohms_law_resistance("-12 V", "-3 A").magnitude == pytest.approx(4.0)
    assert ohms_law_resistance("0 V", "2 A").magnitude == 0.0

    # 100 m copper wire (rho = 1.68e-8 ohm*m), 2.5 mm^2 cross section => 0.672 ohm
    r_wire = wire_resistance("1.68e-8 ohm * m", "100 m", "2.5 mm^2")
    assert str(r_wire.units) == "ohm"
    assert r_wire.magnitude == pytest.approx(0.672)

    # All three power forms agree for V=12 V, I=3 A, R=4 ohm => 36 W
    p_vi = electric_power(voltage="12 V", current="3 A")
    p_ir = electric_power(current="3 A", resistance="4 ohm")
    p_vr = electric_power(voltage="12 V", resistance="4 ohm")
    assert str(p_vi.units) == "watt"
    assert p_vi.magnitude == pytest.approx(36.0)
    assert p_ir.magnitude == pytest.approx(36.0)
    assert p_vr.magnitude == pytest.approx(36.0)

    # 2 A through 50 ohm for 10 min => 200 W * 600 s = 120 kJ
    heat = joule_heating("2 A", "50 ohm", "10 min")
    assert str(heat.units) == "joule"
    assert heat.to("kJ").magnitude == pytest.approx(120.0)


def test_resistor_networks_and_voltage_divider() -> None:
    r_ser = series_resistance(["100 ohm", "2.2 kohm", "0 ohm"])
    assert str(r_ser.units) == "ohm"
    assert r_ser.magnitude == pytest.approx(2300.0)

    r_par = parallel_resistance(["600 ohm", "300 ohm", "200 ohm"])
    assert str(r_par.units) == "ohm"
    assert r_par.magnitude == pytest.approx(100.0)

    v_out = voltage_divider("12 V", "10 kohm", "20 kohm")
    assert str(v_out.units) == "volt"
    assert v_out.magnitude == pytest.approx(8.0)
    assert voltage_divider("12 V", "10 kohm", "0 ohm").magnitude == 0.0


def test_circuits_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        ohms_law_voltage("2 A", "10 C")
    with pytest.raises(pint.DimensionalityError):
        wire_resistance("1.68e-8 ohm", "10 m", "1 mm^2")
    with pytest.raises(pint.DimensionalityError):
        electric_power(voltage="12 V", current="2 C")
    with pytest.raises(pint.DimensionalityError):
        joule_heating("2 A", "5 ohm", "3 m")
    with pytest.raises(pint.DimensionalityError):
        series_resistance(["10 ohm", "5 F"])
    with pytest.raises(pint.DimensionalityError):
        voltage_divider("5 A", "10 ohm", "10 ohm")

    with pytest.raises(ValueError):
        electric_power(voltage="12 V")
    with pytest.raises(ValueError):
        electric_power(voltage="12 V", current="3 A", resistance="4 ohm")
    with pytest.raises(TypeError):
        series_resistance("10 ohm")

    with pytest.raises(PhysicalDomainError):
        ohms_law_voltage("2 A", "-5 ohm")
    with pytest.raises(PhysicalDomainError):
        ohms_law_current("12 V", "0 ohm")
    with pytest.raises(PhysicalDomainError):
        ohms_law_resistance("12 V", "0 A")
    with pytest.raises(PhysicalDomainError):
        ohms_law_resistance("12 V", "-3 A")
    with pytest.raises(PhysicalDomainError):
        wire_resistance("1.68e-8 ohm * m", "10 m", "0 mm^2")
    with pytest.raises(PhysicalDomainError):
        electric_power(voltage="12 V", resistance="0 ohm")
    with pytest.raises(PhysicalDomainError):
        electric_power(current="2 A", resistance="-4 ohm")
    with pytest.raises(PhysicalDomainError):
        joule_heating("2 A", "4 ohm", "-1 s")
    with pytest.raises(PhysicalDomainError):
        series_resistance([])
    with pytest.raises(PhysicalDomainError):
        parallel_resistance(["10 ohm", "0 ohm"])
    with pytest.raises(PhysicalDomainError):
        voltage_divider("5 V", "0 ohm", "0 ohm")


def test_lorentz_force_and_cyclotron_motion() -> None:
    f_lorentz = lorentz_magnetic_force(-ELEMENTARY_CHARGE, "2.0e6 m/s", "500 mT", angle="30 deg")
    e_val = float(ELEMENTARY_CHARGE.to("C").magnitude)
    assert str(f_lorentz.units) == "newton"
    assert f_lorentz.magnitude == pytest.approx(e_val * 2.0e6 * 0.5 * 0.5)
    assert (
        lorentz_magnetic_force(ELEMENTARY_CHARGE, "1e5 m/s", "1 T", angle="180 deg").magnitude
        == 0.0
    )

    f_wire = magnetic_force_on_wire("-5 A", "40 cm", "250 mT")
    assert str(f_wire.units) == "newton"
    assert f_wire.magnitude == pytest.approx(0.5)
    assert magnetic_force_on_wire("5 A", "40 cm", "250 mT", angle="0 deg").magnitude == 0.0

    r_cyc = cyclotron_radius(ELECTRON_MASS, "1.0e6 m/s", -ELEMENTARY_CHARGE, "10 mT")
    m_e = float(ELECTRON_MASS.to("kg").magnitude)
    assert str(r_cyc.units) == "meter"
    assert r_cyc.magnitude == pytest.approx(m_e * 1.0e6 / (e_val * 0.01))

    f_cyc = cyclotron_frequency(ELEMENTARY_CHARGE, "10 mT", ELECTRON_MASS)
    assert str(f_cyc.units) == "hertz"
    assert f_cyc.to("MHz").magnitude == pytest.approx(e_val * 0.01 / (2.0 * math.pi * m_e) / 1e6)


def test_magnetic_fields_and_parallel_wires() -> None:
    mu_0 = float(VACUUM_PERMEABILITY.to("N / A^2").magnitude)

    b_wire = magnetic_field_wire("10 A", "5 cm")
    assert str(b_wire.units) == "tesla"
    assert b_wire.to("uT").magnitude == pytest.approx(mu_0 * 10.0 / (2.0 * math.pi * 0.05) * 1e6)

    b_loop = magnetic_field_loop_center("2 A", "10 cm", turns=50, relative_permeability=1.0)
    assert str(b_loop.units) == "tesla"
    assert b_loop.magnitude == pytest.approx(mu_0 * 50.0 * 2.0 / (2.0 * 0.1))

    b_sol = magnetic_field_solenoid(500, "20 cm", "2 A", relative_permeability=200)
    assert str(b_sol.units) == "tesla"
    assert b_sol.magnitude == pytest.approx(mu_0 * 200.0 * (500.0 / 0.2) * 2.0)

    f_per_l = force_per_length_parallel_wires("10 A", "-10 A", "10 cm")
    assert str(f_per_l.units) == "newton / meter"
    assert f_per_l.magnitude == pytest.approx(mu_0 * 100.0 / (2.0 * math.pi * 0.1))


def test_magnetism_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        lorentz_magnetic_force("1 C", "10 m/s", "1 V")
    with pytest.raises(pint.DimensionalityError):
        magnetic_force_on_wire("2 C", "1 m", "0.5 T")
    with pytest.raises(pint.DimensionalityError):
        cyclotron_radius("1 kg", "10 m/s", "1 C", "10 gauss")
    with pytest.raises(pint.DimensionalityError):
        magnetic_field_solenoid("500 m", "20 cm", "2 A")
    with pytest.raises(pint.DimensionalityError):
        force_per_length_parallel_wires("5 A", "5 A", "10 s")

    with pytest.raises(PhysicalDomainError):
        lorentz_magnetic_force("1 C", "-10 m/s", "1 T")
    with pytest.raises(PhysicalDomainError):
        lorentz_magnetic_force("1 C", "10 m/s", "-1 T")
    with pytest.raises(PhysicalDomainError):
        magnetic_force_on_wire("2 A", "-1 m", "0.5 T")
    with pytest.raises(PhysicalDomainError):
        cyclotron_radius(ELECTRON_MASS, "10 m/s", "0 C", "1 T")
    with pytest.raises(PhysicalDomainError):
        cyclotron_radius(ELECTRON_MASS, "10 m/s", ELEMENTARY_CHARGE, "0 T")
    with pytest.raises(PhysicalDomainError):
        cyclotron_frequency(ELEMENTARY_CHARGE, "1 T", "0 kg")
    with pytest.raises(PhysicalDomainError):
        magnetic_field_wire("10 A", "0 m")
    with pytest.raises(PhysicalDomainError):
        magnetic_field_loop_center("2 A", "10 cm", turns=0)
    with pytest.raises(PhysicalDomainError):
        magnetic_field_solenoid(500, "20 cm", "2 A", relative_permeability=0)
    with pytest.raises(PhysicalDomainError):
        force_per_length_parallel_wires("5 A", "5 A", "-1 cm")


def test_capacitors_dielectrics_and_energy() -> None:
    c_val = capacitance("240 uC", "12 V")
    assert str(c_val.units) == "farad"
    assert c_val.to("uF").magnitude == pytest.approx(20.0)
    assert capacitance("-24 uC", "-12 V").to("uF").magnitude == pytest.approx(2.0)
    assert capacitance("0 C", "12 V").magnitude == 0.0

    q_val = capacitor_charge("100 uF", "50 V")
    assert str(q_val.units) == "coulomb"
    assert q_val.to("mC").magnitude == pytest.approx(5.0)

    v_val = capacitor_voltage("5 mC", "100 uF")
    assert str(v_val.units) == "volt"
    assert v_val.magnitude == pytest.approx(50.0)

    eps_0 = float(VACUUM_PERMITTIVITY.to("F / m").magnitude)
    c_plate = parallel_plate_capacitance("10 cm^2", "0.1 mm", relative_permittivity=4.5)
    assert str(c_plate.units) == "farad"
    assert c_plate.magnitude == pytest.approx(eps_0 * 4.5 * 10e-4 / 0.1e-3)

    # All 3 capacitor energy forms agree: C = 100 uF, V = 50 V, Q = 5 mC => U = 0.125 J
    u_cv = capacitor_energy(capacitance="100 uF", voltage="50 V")
    u_cq = capacitor_energy(capacitance="100 uF", charge="5 mC")
    u_qv = capacitor_energy(charge="5 mC", voltage="50 V")
    assert str(u_cv.units) == "joule"
    assert u_cv.magnitude == pytest.approx(0.125)
    assert u_cq.magnitude == pytest.approx(0.125)
    assert u_qv.magnitude == pytest.approx(0.125)


def test_capacitor_networks_and_rc_transients() -> None:
    c_ser = series_capacitance(["6 uF", "3 uF"])
    assert str(c_ser.units) == "farad"
    assert c_ser.to("uF").magnitude == pytest.approx(2.0)

    c_par = parallel_capacitance(["100 nF", "400 nF", "0 pF"])
    assert str(c_par.units) == "farad"
    assert c_par.to("nF").magnitude == pytest.approx(500.0)

    tau = rc_time_constant("10 kohm", "100 uF")
    assert str(tau.units) == "second"
    assert tau.magnitude == pytest.approx(1.0)

    v_chg_0 = rc_charging_voltage("12 V", "0 s", "10 kohm", "100 uF")
    v_chg_1tau = rc_charging_voltage("12 V", "1 s", "10 kohm", "100 uF")
    assert v_chg_0.magnitude == 0.0
    assert v_chg_1tau.magnitude == pytest.approx(12.0 * (1.0 - math.exp(-1.0)))

    v_dis_1tau = rc_discharging_voltage("12 V", "1000 ms", "10 kohm", "100 uF")
    assert v_dis_1tau.magnitude == pytest.approx(12.0 * math.exp(-1.0))


def test_capacitance_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        capacitance("10 C", "5 A")
    with pytest.raises(pint.DimensionalityError):
        parallel_plate_capacitance("10 m", "1 mm")
    with pytest.raises(pint.DimensionalityError):
        rc_time_constant("10 ohm", "10 H")
    with pytest.raises(pint.DimensionalityError):
        series_capacitance(["10 uF", "5 ohm"])

    with pytest.raises(ValueError):
        capacitor_energy(capacitance="10 uF")
    with pytest.raises(ValueError):
        capacitor_energy(capacitance="10 uF", voltage="5 V", charge="50 uC")
    with pytest.raises(TypeError):
        parallel_capacitance("10 uF")

    with pytest.raises(PhysicalDomainError):
        capacitance("10 uC", "0 V")
    with pytest.raises(PhysicalDomainError):
        capacitance("10 uC", "-5 V")
    with pytest.raises(PhysicalDomainError):
        capacitor_charge("-10 uF", "5 V")
    with pytest.raises(PhysicalDomainError):
        capacitor_voltage("10 uC", "0 uF")
    with pytest.raises(PhysicalDomainError):
        parallel_plate_capacitance("10 cm^2", "0 mm")
    with pytest.raises(PhysicalDomainError):
        parallel_plate_capacitance("10 cm^2", "1 mm", relative_permittivity=0)
    with pytest.raises(PhysicalDomainError):
        capacitor_energy(capacitance="0 uF", charge="5 uC")
    with pytest.raises(PhysicalDomainError):
        capacitor_energy(charge="5 uC", voltage="-10 V")
    with pytest.raises(PhysicalDomainError):
        series_capacitance(["10 uF", "0 uF"])
    with pytest.raises(PhysicalDomainError):
        rc_charging_voltage("12 V", "-1 s", "10 kohm", "100 uF")
    with pytest.raises(PhysicalDomainError):
        rc_discharging_voltage("12 V", "1 s", "0 ohm", "100 uF")


def test_flux_faraday_and_motional_emf() -> None:
    phi = magnetic_flux("0.5 T", "20 cm^2", angle="60 deg")
    assert str(phi.units) == "weber"
    assert phi.to("mWb").magnitude == pytest.approx(0.5)
    assert magnetic_flux("0.5 T", "20 cm^2", angle="90 deg").magnitude == 0.0

    emf = faraday_emf("5 mWb", "20 ms", turns=100)
    assert str(emf.units) == "volt"
    assert emf.magnitude == pytest.approx(-25.0)
    assert faraday_emf("0 Wb", "20 ms", turns=50).magnitude == 0.0

    emf_mot = motional_emf("0.5 T", "40 cm", "10 m/s")
    assert str(emf_mot.units) == "volt"
    assert emf_mot.magnitude == pytest.approx(2.0)
    assert motional_emf("0.5 T", "40 cm", "10 m/s", angle="0 deg").magnitude == 0.0


def test_inductors_rl_lc_and_transformers() -> None:
    mu_0 = float(VACUUM_PERMEABILITY.to("N / A^2").magnitude)
    l_sol = solenoid_inductance(500, "10 cm^2", "20 cm", relative_permeability=1.0)
    assert str(l_sol.units) == "henry"
    assert l_sol.magnitude == pytest.approx(mu_0 * (500**2) * 10e-4 / 0.2)

    u_l = inductor_energy("10 mH", "-3 A")
    assert str(u_l.units) == "joule"
    assert u_l.to("mJ").magnitude == pytest.approx(45.0)

    l_ser = series_inductance(["10 mH", "15 mH", "0 H"])
    assert str(l_ser.units) == "henry"
    assert l_ser.to("mH").magnitude == pytest.approx(25.0)

    l_par = parallel_inductance(["60 mH", "30 mH"])
    assert str(l_par.units) == "henry"
    assert l_par.to("mH").magnitude == pytest.approx(20.0)

    tau_rl = rl_time_constant("50 mH", "10 ohm")
    assert str(tau_rl.units) == "second"
    assert tau_rl.to("ms").magnitude == pytest.approx(5.0)

    f_lc = lc_resonant_frequency("10 mH", "10 uF")
    assert str(f_lc.units) == "hertz"
    assert f_lc.magnitude == pytest.approx(1.0 / (2.0 * math.pi * math.sqrt(10e-3 * 10e-6)))

    v_s = transformer_voltage("240 V", 1000, 50)
    i_s = transformer_current("500 mA", 1000, 50)
    assert str(v_s.units) == "volt"
    assert v_s.magnitude == pytest.approx(12.0)
    assert str(i_s.units) == "ampere"
    assert i_s.magnitude == pytest.approx(10.0)


def test_induction_validation_errors() -> None:
    with pytest.raises(pint.DimensionalityError):
        magnetic_flux("0.5 T", "10 m")
    with pytest.raises(pint.DimensionalityError):
        faraday_emf("5 mT", "10 ms")
    with pytest.raises(pint.DimensionalityError):
        inductor_energy("10 mF", "2 A")
    with pytest.raises(pint.DimensionalityError):
        rl_time_constant("10 mH", "5 F")
    with pytest.raises(pint.DimensionalityError):
        lc_resonant_frequency("10 mH", "5 ohm")

    with pytest.raises(PhysicalDomainError):
        magnetic_flux("0.5 T", "-1 m^2")
    with pytest.raises(PhysicalDomainError):
        faraday_emf("5 mWb", "0 s")
    with pytest.raises(PhysicalDomainError):
        faraday_emf("5 mWb", "1 s", turns=0)
    with pytest.raises(PhysicalDomainError):
        motional_emf("-0.5 T", "1 m", "2 m/s")
    with pytest.raises(PhysicalDomainError):
        solenoid_inductance(100, "1 cm^2", "0 m")
    with pytest.raises(PhysicalDomainError):
        inductor_energy("-10 mH", "2 A")
    with pytest.raises(PhysicalDomainError):
        parallel_inductance(["10 mH", "0 H"])
    with pytest.raises(PhysicalDomainError):
        rl_time_constant("10 mH", "0 ohm")
    with pytest.raises(PhysicalDomainError):
        lc_resonant_frequency("0 H", "10 uF")
    with pytest.raises(PhysicalDomainError):
        lc_resonant_frequency("10 mH", "0 F")
    with pytest.raises(PhysicalDomainError):
        transformer_voltage("120 V", 0, 10)
    with pytest.raises(PhysicalDomainError):
        transformer_current("1 A", 10, -5)


def test_electromagnetism_units_external_registry_and_exports() -> None:
    # Battery capacity in mAh and energy in kWh / eV
    c_bank = capacitance("500 mAh", "3.7 V")
    assert c_bank.to("F").magnitude == pytest.approx(1800.0 / 3.7)

    heat_kwh = joule_heating("10 A", "20 ohm", "2 hour")
    assert heat_kwh.to("kWh").magnitude == pytest.approx(4.0)

    # External Pint registry compatibility
    ext_ureg = pint.UnitRegistry()
    f_ext = coulomb_force(ext_ureg.Quantity(1, "uC"), ext_ureg.Quantity(2, "uC"), "10 cm")
    assert f_ext.magnitude == pytest.approx(coulomb_force("1 uC", "2 uC", "10 cm").magnitude)

    # Non-finite rejection
    with pytest.raises(PhysicalDomainError):
        electric_field_point_charge("nan C", "1 m")
    with pytest.raises(PhysicalDomainError):
        ohms_law_current("inf V", "10 ohm")

    # Package __all__ completeness across all 5 submodules
    for submodule in (
        electromagnetism.electrostatics,
        electromagnetism.circuits,
        electromagnetism.capacitance_module,
        electromagnetism.magnetism,
        electromagnetism.induction,
    ):
        for symbol in submodule.__all__:
            assert symbol in electromagnetism.__all__
            assert hasattr(electromagnetism, symbol)
