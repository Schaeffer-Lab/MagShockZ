"""Pure functions for temperature profile and anisotropy analysis.

Temperature is the second central moment of the phase-space distribution
along one momentum axis, in simulation units (m_e c^2):
    T = |rqm| * <(u - <u>)^2>
where u = p / (m_e * c) is the OSIRIS normalised momentum and |rqm| (= m/q,
the OSIRIS mass-per-charge = sim macroparticle mass in units of m_e) converts
from electron-mass units to the species mass.

Directions:
    p1 — along the shock propagation direction (shock normal)
    p2 — transverse (in-plane perpendicular to shock normal)
"""

import numpy as np

import moments


def temperature_profile(
    phase_space,
    rqm: float,
    momentum_axis: str,
) -> np.ndarray:
    """Return temperature profile T(x) in simulation units (m_e c^2).

    Parameters
    ----------
    phase_space : osh5def.H5Data
        Phase-space distribution at a single dump.
    rqm : float
        OSIRIS rqm parameter (m/q, mass-per-charge, in units of m_e/e). |rqm|
        equals the sim macroparticle mass in units of m_e and converts the
        thermal spread from electron-mass units to the species mass.
    momentum_axis : str
        Name of the momentum axis to integrate over ("p1" or "p2").
    """
    vth2 = moments.moment(phase_space, axis=momentum_axis, order=2)
    return abs(rqm) * vth2


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    floor: float = 1e-10,
) -> np.ndarray:
    """Element-wise ratio; returns nan where |denominator| < floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denominator) > floor, numerator / denominator, np.nan)


def region_averages(
    arr: np.ndarray,
    x_axis: np.ndarray,
    x_shock: float,
    x_downstream_start: float,
) -> tuple[float, float]:
    """Return (upstream_mean, downstream_mean) of arr across the shock."""
    downstream = (x_axis >= x_downstream_start) & (x_axis <= x_shock)
    upstream = x_axis > x_shock
    return float(np.nanmean(arr[upstream])), float(np.nanmean(arr[downstream]))
