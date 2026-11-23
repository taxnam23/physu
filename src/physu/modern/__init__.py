"""Modern physics formulas and utilities for physu."""

from __future__ import annotations

from . import atomic, nuclear, photons, quantum, relativity
from .atomic import (
    bohr_energy_level,
    bohr_orbit_radius,
    bohr_transition_energy,
    rydberg_wavelength,
)
from .nuclear import (
    decay_constant,
    half_life_from_decay_constant,
    mass_energy_equivalence,
    radioactive_activity,
    radioactive_remaining,
    radiometric_age,
)
from .photons import (
    compton_wavelength_shift,
    photoelectric_max_kinetic_energy,
    photoelectric_stopping_potential,
    photoelectric_threshold_frequency,
    photon_energy,
    photon_momentum,
    wien_peak_wavelength,
)
from .quantum import (
    de_broglie_wavelength,
    de_broglie_wavelength_particle,
    heisenberg_min_energy_uncertainty,
    heisenberg_min_momentum_uncertainty,
    infinite_well_energy,
)
from .relativity import (
    energy_momentum_relation,
    length_contraction,
    lorentz_factor,
    relativistic_kinetic_energy,
    relativistic_momentum,
    relativistic_total_energy,
    relativistic_velocity_addition,
    rest_energy,
    time_dilation,
)

__all__ = [
    "atomic",
    "bohr_energy_level",
    "bohr_orbit_radius",
    "bohr_transition_energy",
    "compton_wavelength_shift",
    "de_broglie_wavelength",
    "de_broglie_wavelength_particle",
    "decay_constant",
    "energy_momentum_relation",
    "half_life_from_decay_constant",
    "heisenberg_min_energy_uncertainty",
    "heisenberg_min_momentum_uncertainty",
    "infinite_well_energy",
    "length_contraction",
    "lorentz_factor",
    "mass_energy_equivalence",
    "nuclear",
    "photoelectric_max_kinetic_energy",
    "photoelectric_stopping_potential",
    "photoelectric_threshold_frequency",
    "photon_energy",
    "photon_momentum",
    "photons",
    "quantum",
    "radioactive_activity",
    "radioactive_remaining",
    "radiometric_age",
    "relativistic_kinetic_energy",
    "relativistic_momentum",
    "relativistic_total_energy",
    "relativistic_velocity_addition",
    "relativity",
    "rest_energy",
    "rydberg_wavelength",
    "time_dilation",
    "wien_peak_wavelength",
]
