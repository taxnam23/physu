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

## Quickstart

```python
import pint
from physu import PhysicalDomainError, ureg
from physu.constants import STANDARD_GRAVITY
from physu.kinematics import (
    accelerated_displacement,
    centripetal_acceleration,
    displacement,
    projectile_range,
    time_to_height,
)

# 1D Uniform Motion (MRU)
dx = displacement("72 km/h", "10 s")
print(dx)            # 200.0 meter
print(dx.to("km"))   # 0.2 kilometer

# 1D Uniformly Accelerated Motion (MRUA)
dx_acc = accelerated_displacement(10 * (ureg.m / ureg.s), "2 m/s^2", "5 s")
print(dx_acc)        # 75.0 meter

# Vertical Motion (returns both times when reaching 15 m: going up & coming down)
times = time_to_height("20 m/s", "15 m", gravity="10 m/s^2")
print(times)         # (Quantity(1.0, "second"), Quantity(3.0, "second"))

# 2D Projectile Motion
rng = projectile_range("20 m/s", "45 deg", gravity=STANDARD_GRAVITY)
print(round(rng, 2)) # 40.79 meter

# Uniform Circular Motion (UCM)
ac = centripetal_acceleration("2 m", angular_velocity="60 rpm")
print(round(ac, 2))  # 78.96 meter / second ** 2

# Catching dimensional & physical errors
try:
    displacement("10 kg", "2 s")
except pint.DimensionalityError as exc:
    print("Dimension error:", exc)

try:
    time_to_height("20 m/s", "25 m", gravity="10 m/s^2")
except PhysicalDomainError as exc:
    print("Domain error:", exc)
```

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

## Development

```bash
pytest
ruff check src tests
mypy src
```

## License

[MIT](LICENSE)
