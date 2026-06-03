"""Pure-function energy partition analysis for OSIRIS shock simulations.

All functions operate on raw numpy arrays so they are testable without OSIRIS
I/O. Data loading and grid extraction happen in the calling script.

All energy densities are in OSIRIS simulation units (n_0 m_e c^2).
"""

import numpy as np

import moments
from analysis_utils import region_masks


def species_energy_profiles(
    phase_space,
    rqm: float,
    v_shock: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_ram, u_th) in simulation units for one species at one timestep.

    Parameters
    ----------
    phase_space : osh5def.H5Data
        Phase-space distribution f(p1, x1) at a single dump.
    rqm : float
        OSIRIS rqm parameter (m/q, i.e. mass-per-charge, in units of m_e/e).
        |rqm| equals the sim macroparticle mass in units of m_e, so it scales
        the kinetic energy from electron-mass units to the species mass;
        for electrons |rqm|=1 so no correction is needed.
    v_shock : float
        Shock velocity in units of c; boosts bulk flow to shock rest frame.
    """
    mass_factor = abs(rqm)

    n_norm = np.abs(moments.moment(phase_space, axis="p1", order=0))
    u_norm = moments.moment(phase_space, axis="p1", order=1) - v_shock
    vth2_norm = np.maximum(moments.moment(phase_space, axis="p1", order=2), 0.0)

    u_ram = 0.5 * n_norm * u_norm**2 * mass_factor
    u_th = n_norm * vth2_norm * mass_factor
    return u_ram, u_th


def field_energy_profiles(
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    x_field: np.ndarray,
    x_shock: float,
    field_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_B, u_E) in simulation units on the field grid.

    In OSIRIS units the EM energy densities are B^2/2 and E^2/2
    (normalized to n_0 m_e c^2), analogous to SI eps0*E^2/2 and B^2/(2 mu0)
    once the field normalization B_0 = m_e c omega_pe / e is substituted.

    Parameters
    ----------
    b1, b2, b3, e1, e2, e3 : np.ndarray
        OSIRIS normalised field arrays on the field grid.
    x_field : np.ndarray
        Spatial coordinate of the field grid [c/ωpe].
    x_shock : float
        Shock position [c/ωpe]; used only when field_mode="delta".
    field_mode : {"full", "delta"}
        "full" uses total B²; "delta" subtracts the upstream mean B first.
    """
    E2 = np.square(e1) + np.square(e2) + np.square(e3)

    if field_mode == "delta":
        upstream = x_field > x_shock
        b1_bg = np.nanmean(b1[upstream]) if upstream.any() else 0.0
        b2_bg = np.nanmean(b2[upstream]) if upstream.any() else 0.0
        b3_bg = np.nanmean(b3[upstream]) if upstream.any() else 0.0
        B2 = (
            np.square(b1 - b1_bg)
            + np.square(b2 - b2_bg)
            + np.square(b3 - b3_bg)
        )
    else:
        B2 = np.square(b1) + np.square(b2) + np.square(b3)

    return 0.5 * B2, 0.5 * E2


def partition_by_region(
    u_ram: np.ndarray,
    u_th: np.ndarray,
    u_B: np.ndarray,
    u_E: np.ndarray,
    x_axis: np.ndarray,
    x_shock: float,
    x_downstream_start: float,
) -> dict:
    """Split energy densities into upstream and downstream mean values.

    Parameters
    ----------
    u_ram, u_th, u_B, u_E : np.ndarray
        Energy density profiles on a common spatial grid, in simulation units.
    x_axis : np.ndarray
        Spatial coordinate grid [c/ωpe].
    x_shock : float
        Shock position; upstream is x > x_shock.
    x_downstream_start : float
        Left boundary of downstream region.

    Returns
    -------
    dict with keys "upstream" and "downstream", each a dict of channel ->
    mean energy density (simulation units).
    """
    upstream, downstream = region_masks(x_axis, x_shock, x_downstream_start)

    if not downstream.any() or not upstream.any():
        raise ValueError(
            f"Empty region mask — check x_shock={x_shock:.1f} and "
            f"x_downstream_start={x_downstream_start:.1f} vs "
            f"x_axis=[{x_axis.min():.1f}, {x_axis.max():.1f}]"
        )

    def _side(mask):
        return {
            "ram": float(np.nanmean(u_ram[mask])),
            "thermal": float(np.nanmean(u_th[mask])),
            "B_field": float(np.nanmean(u_B[mask])),
            "E_field": float(np.nanmean(u_E[mask])),
        }

    return {"upstream": _side(upstream), "downstream": _side(downstream)}
