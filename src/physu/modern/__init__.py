"""Modern physics formulas and utilities for physu."""

from __future__ import annotations

from . import photons, relativity
from .photons import (
    compton_wavelength_shift,
    photoelectric_max_kinetic_energy,
    photoelectric_stopping_potential,
    photoelectric_threshold_frequency,
    photon_energy,
    photon_momentum,
    wien_peak_wavelength,
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
    "compton_wavelength_shift",
    "energy_momentum_relation",
    "length_contraction",
    "lorentz_factor",
    "photoelectric_max_kinetic_energy",
    "photoelectric_stopping_potential",
    "photoelectric_threshold_frequency",
    "photon_energy",
    "photon_momentum",
    "photons",
    "relativistic_kinetic_energy",
    "relativistic_momentum",
    "relativistic_total_energy",
    "relativistic_velocity_addition",
    "relativity",
    "rest_energy",
    "time_dilation",
    "wien_peak_wavelength",
]
