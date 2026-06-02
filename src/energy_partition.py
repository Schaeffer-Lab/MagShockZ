"""Pure-function energy partition analysis for OSIRIS shock simulations.

All functions operate on raw numpy arrays so they are testable without OSIRIS
I/O. Data loading and grid extraction happen in the calling script.
"""

from dataclasses import dataclass

import astropy.constants
import astropy.units
import numpy as np

import moments


@dataclass
class FieldNorm:
    """OSIRIS EM field normalisation units derived from omega_pe."""
    B_unit_SI: astropy.units.Quantity  # Tesla
    E_unit_SI: astropy.units.Quantity  # V/m


def field_normalization(omega_pe: astropy.units.Quantity) -> FieldNorm:
    """Compute OSIRIS field normalisation from the electron plasma frequency.

    Both B and E normalise to m_e * c * omega_pe / e, expressed in
    Gaussian-convenient pairs (Gauss, GV/cm).
    """
    omega_si = (omega_pe / astropy.units.rad).to(1 / astropy.units.s).value
    B_SI = (5.681e-8 * omega_si * astropy.units.Gauss).to(astropy.units.T)
    E_SI = (1.704e-14 * omega_si * astropy.units.GV / astropy.units.cm).to(
        astropy.units.V / astropy.units.m
    )
    return FieldNorm(B_unit_SI=B_SI, E_unit_SI=E_SI)


def species_energy_profiles(
    phase_space,
    mass: astropy.units.Quantity,
    v_shock: float,
    n0_si: astropy.units.Quantity,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_ram, u_th) in J/m³ for one species at one timestep.

    Parameters
    ----------
    phase_space : osh5def.H5Data
        Phase-space distribution f(p1, x1) at a single dump.
    mass : astropy Quantity
        Particle rest mass (e.g. ``abs(sim.rqm) * astropy.constants.m_e``).
    v_shock : float
        Shock velocity in units of c; boosts bulk flow to shock rest frame.
    n0_si : astropy Quantity
        Normalisation number density in SI (m^-3).
    """
    c = astropy.constants.c

    n_norm = np.abs(moments.moment(phase_space, axis="p1", order=0))
    u_norm = moments.moment(phase_space, axis="p1", order=1) - v_shock
    vth2_norm = np.maximum(moments.moment(phase_space, axis="p1", order=2), 0.0)

    n_si = n_norm * n0_si
    u_ram = (0.5 * n_si * mass * (u_norm * c) ** 2).to(
        astropy.units.J / astropy.units.m**3
    )
    u_th = (n_si * vth2_norm * mass * c**2).to(astropy.units.J / astropy.units.m**3)
    return u_ram.value, u_th.value


def field_energy_profiles(
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    norm: FieldNorm,
    x_field: np.ndarray,
    x_shock: float,
    field_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_B, u_E) in J/m³ on the field grid.

    Parameters
    ----------
    b1, b2, b3, e1, e2, e3 : np.ndarray
        OSIRIS normalised field arrays on the field grid.
    norm : FieldNorm
        SI normalisation from field_normalization().
    x_field : np.ndarray
        Spatial coordinate of the field grid [c/ωpe].
    x_shock : float
        Shock position [c/ωpe]; used only when field_mode="delta".
    field_mode : {"full", "delta"}
        "full" uses total B²; "delta" subtracts the upstream mean B first.
    """
    mu0 = astropy.constants.mu0
    eps0 = astropy.constants.eps0

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

    u_E = (0.5 * eps0 * E2 * norm.E_unit_SI**2).to(
        astropy.units.J / astropy.units.m**3
    ).value
    u_B = (0.5 * B2 * norm.B_unit_SI**2 / mu0).to(
        astropy.units.J / astropy.units.m**3
    ).value
    return u_B, u_E


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
        Energy density profiles on a common spatial grid, in J/m³.
    x_axis : np.ndarray
        Spatial coordinate grid [c/ωpe].
    x_shock : float
        Shock position; upstream is x > x_shock.
    x_downstream_start : float
        Left boundary of downstream region.

    Returns
    -------
    dict with keys "upstream" and "downstream", each a dict of channel ->
    mean energy density (J/m³).
    """
    downstream = (x_axis >= x_downstream_start) & (x_axis <= x_shock)
    upstream = x_axis > x_shock

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
