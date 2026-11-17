# physu

A modular Python physics library with strict unit and dimensional validation powered by [Pint](https://pint.readthedocs.io/).

## Features

- **Strict dimensional safety:** Rejects incompatible dimensions (such as mass where velocity is expected) via `pint.DimensionalityError`.
- **Physical domain validation:** Rejects non-physical inputs (negative time, zero radius, unreachable trajectories, `NaN`, or non-finite values) via `PhysicalDomainError` (a `ValueError` subclass).
- **Centralized `UnitRegistry`:** Shares a single `physu.ureg` across all modules and normalizes strings (`"72 km/h"`), `physu.ureg` quantities, and quantities from external Pint registries.
- **Clean SI outputs:** Calculations return normalized SI `pint.Quantity` objects that can be converted to any compatible unit with `.to(...)`.

## Installation

```bash
pip install physu
```

For local development (including `pytest`, `ruff`, and `mypy`):

```bash
pip install -e ".[dev]"
```

Requires Python `>= 3.10`.

## Examples

Runnable scripts covering every module and submodule live in [examples/](examples/):

- [units_and_validation.py](examples/units_and_validation.py) — units, constants, validation helpers, and error handling
- [kinematics_1d.py](examples/kinematics_1d.py) — uniform, accelerated (MRUA/SUVAT), and vertical motion
- [kinematics_2d_and_circular.py](examples/kinematics_2d_and_circular.py) — 2D projectile and uniform circular motion
- [dynamics_forces_and_friction.py](examples/dynamics_forces_and_friction.py) — Newton's laws, inclined planes, and friction
- [dynamics_elasticity_momentum_gravitation.py](examples/dynamics_elasticity_momentum_gravitation.py) — Hooke's law, momentum/collisions, and gravitation

## Modules

- **`physu.units`**: Centralized `ureg`, `Quantity`, `to_quantity`, `check_dimension`, `convert_to`, `to_si`, and `to_unit`.
- **`physu.exceptions`**: `PhysuError` and `PhysicalDomainError`.
- **`physu.validation`**: `validate_quantity`, `require_non_negative`, `require_positive`, `require_non_zero`, and `require_range`.
- **`physu.constants`**: Fundamental SI / CODATA 2022 constants (`STANDARD_GRAVITY`, `SPEED_OF_LIGHT`, `GRAVITATIONAL_CONSTANT`, `PLANCK_CONSTANT`, `ELEMENTARY_CHARGE`, `BOLTZMANN_CONSTANT`, `AVOGADRO_CONSTANT`, `MOLAR_GAS_CONSTANT`, `STANDARD_ATMOSPHERE`, `ELECTRON_MASS`, `PROTON_MASS`, `VACUUM_PERMITTIVITY`, `VACUUM_PERMEABILITY`).
- **`physu.kinematics`**:
  - `uniform`: MRU (`displacement`, `position`, `average_velocity`, `travel_time`)
  - `accelerated`: MRUA / SUVAT (`final_velocity`, `displacement`, `position`, `displacement_from_velocities`, `acceleration`, `torricelli_velocity`, `time_to_velocity`, `time_to_displacement`)
  - `vertical`: Free fall and vertical throw (`vertical_position`, `vertical_velocity`, `free_fall_distance`, `free_fall_speed`, `free_fall_time`, `time_to_peak`, `max_height`, `time_to_height`)
  - `projectile`: 2D projectile motion (`velocity_components`, `projectile_flight_time`, `projectile_max_height`, `projectile_range`, `projectile_position`, `projectile_velocity`)
  - `circular`: Uniform circular motion (`tangential_speed`, `angular_velocity_from_speed`, `centripetal_acceleration`, `period_from_frequency`, `frequency_from_period`, `angular_velocity_from_period`, `angular_velocity_from_frequency`, `period_from_angular_velocity`, `frequency_from_angular_velocity`, `angular_displacement`, `arc_length`)
- **`physu.dynamics`**:
  - `newton`: Newton's laws, weight, and inclined planes (`force`, `acceleration`, `mass`, `weight`, `inclined_parallel_force`, `inclined_normal_force`)
  - `friction`: Static and kinetic friction (`max_static_friction`, `kinetic_friction`, `friction_coefficient`, `normal_force`, `critical_incline_angle`)
  - `elasticity`: Hooke's law and springs (`spring_force`, `spring_force_magnitude`, `spring_constant`, `spring_displacement`, `series_spring_constant`, `parallel_spring_constant`)
  - `momentum`: Linear momentum, impulse, and 1D collisions (`linear_momentum`, `impulse`, `momentum_change`, `average_force`, `collision_velocities`, `elastic_collision_velocities`, `inelastic_collision_velocity`)
  - `gravitation`: Universal gravitation and circular dynamics (`gravitational_force`, `gravitational_field_strength`, `orbital_speed`, `centripetal_force`)
- **`physu.energy`**:
  - `work`: Mechanical work and work-energy theorem (`mechanical_work`, `net_work`, `gravitational_work`, `elastic_work`)

## Development

```bash
pytest
ruff check src tests
mypy src
```

## License

[MIT](LICENSE)
