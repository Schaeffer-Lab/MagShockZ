"""Generation of the WarpX heater-driven piston deck.

``units``   -- primaries -> every derived scale (the single source of truth)
``config``  -- load / validate / freeze a run's ``config.yaml``
``deck``    -- render the ParmParse deck, and parse it back to prove it still means it

The comparison against FLASH is post-processing and lives in
``magshockz.analysis.warpx``.
"""

from __future__ import annotations

from . import units
from .units import (
    ALUMINIUM_6,
    SILICON_14,
    DeckScales,
    FlashReference,
    Upstream,
    derive,
    mass_per_charge,
    reduce_mass,
    theta,
)

__all__ = [
    "ALUMINIUM_6",
    "SILICON_14",
    "DeckScales",
    "FlashReference",
    "Upstream",
    "derive",
    "mass_per_charge",
    "reduce_mass",
    "theta",
    "units",
]
