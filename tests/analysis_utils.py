"""Minimal analysis_utils stub for test environments.

Provides only the symbols imported by the pure-function modules under test
(energy_partition.py, temperature_anisotropy.py) so the full OSIRIS/astropy
stack is not required in CI.
"""

import numpy as np


def region_masks(x, x_shock: float, x_downstream_start: float):
    """Boolean (upstream, downstream) masks — mirrors the real implementation."""
    x = np.asarray(x)
    upstream = x > x_shock
    downstream = (x >= x_downstream_start) & (x <= x_shock)
    return upstream, downstream
