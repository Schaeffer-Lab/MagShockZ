"""Shared core for the FLASH -> OSIRIS initialization entrypoints.

Both the python-coupled (full-2D) and math-function (1D / quasi-1D) writers
import from here so the plasma physics lives in exactly one place.
"""

from .normalizations import Normalizations, compute_norms, reference_check_lines
from .run_params import (
    ParticleLoad,
    cfl_dt,
    estimate_particle_load,
    max_tile_cells,
    ndump,
    osiris_quoted_list,
    tile_numbers,
)

__all__ = [
    "Normalizations",
    "compute_norms",
    "reference_check_lines",
    "ParticleLoad",
    "cfl_dt",
    "estimate_particle_load",
    "max_tile_cells",
    "ndump",
    "osiris_quoted_list",
    "tile_numbers",
]
