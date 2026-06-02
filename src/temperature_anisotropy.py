"""Pure functions for temperature profile and anisotropy analysis.

Temperature is the second central moment of the phase-space distribution
along one momentum axis, scaled by rest-mass energy:
    T = m * c^2 * <(u - <u>)^2>  [eV]
where u = p / (m_e * c) is the OSIRIS normalised momentum.

Directions:
    p1 — along the shock propagation direction (shock normal)
    p2 — transverse (in-plane perpendicular to shock normal)
"""

import astropy.constants
import astropy.units
import numpy as np

import moments


def temperature_profile(
    phase_space,
    mass: astropy.units.Quantity,
    momentum_axis: str,
) -> np.ndarray:
    """Return temperature profile T(x) in eV.

    Parameters
    ----------
    phase_space : osh5def.H5Data
        Phase-space distribution at a single dump.
    mass : astropy Quantity
        Particle rest mass.
    momentum_axis : str
        Name of the momentum axis to integrate over ("p1" or "p2").
    """
    c = astropy.constants.c
    vth2 = moments.moment(phase_space, axis=momentum_axis, order=2)
    return (vth2 * mass * c**2).to(astropy.units.eV).value


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
