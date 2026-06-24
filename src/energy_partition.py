"""Pure-function energy partition analysis for OSIRIS shock simulations.

All functions operate on raw numpy arrays so they are testable without OSIRIS
I/O. Data loading and grid extraction happen in the calling script.

All energy densities are in OSIRIS simulation units (n_0 m_e c^2).
"""

import numpy as np

import moments
from analysis_utils import region_masks


def _momentum_axis_name(phase_space) -> str:
    """Name of the momentum axis of a pNx1 phase space (the non-spatial axis)."""
    for ax in phase_space.axes:
        if ax.name.startswith("p"):
            return ax.name
    raise ValueError(
        "Phase space has no momentum axis (no axis whose name starts with 'p')"
    )


def species_energy_profiles(
    phase_space,
    rqm: float,
    v_shock: float,
    perp_phase_spaces=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_ram, u_th) in simulation units for one species at one timestep.

    Energy densities are in OSIRIS units (n_0 m_e c^2).  OSIRIS normalises each
    species' momentum to its own mass (u = p / (m_species c) = v/c non-rel.), so
    the variance returned by the 2nd moment is sigma^2 = uth^2 = T / (m_s c^2)
    and the bulk velocity is directly comparable across species (and to v_shock).
    The per-particle kinetic energy is then 0.5 * m_s c^2 * u^2; multiplying by
    |rqm| = m_s / m_e converts to m_e c^2 units.  Hence both channels carry the
    0.5 of ``KE = 1/2 m v^2`` and the thermal channel sums over the available
    momentum directions to give (3/2) n T_iso when isotropic.

    Parameters
    ----------
    phase_space : osh5def.H5Data
        Shock-normal phase-space distribution f(p1, x1) at a single dump.  Sets
        the number density, the (shock-frame) bulk-flow ram energy, and the p1
        thermal contribution.
    rqm : float
        OSIRIS rqm parameter (m/q, i.e. mass-per-charge, in units of m_e/e).
        |rqm| equals the sim macroparticle mass in units of m_e, so it scales
        the kinetic energy from electron-mass units to the species mass;
        for electrons |rqm|=1 so no correction is needed.
    v_shock : float
        Shock velocity in units of c; boosts bulk flow to shock rest frame.
    perp_phase_spaces : iterable of osh5def.H5Data, optional
        Transverse phase spaces (e.g. p2x1, p3x1) when those diagnostics are
        output.  Each contributes its velocity-space variance to the thermal
        energy, so u_th becomes the full multi-direction thermal energy.  The
        momentum axis of each is detected automatically; the variance is taken
        about its own per-direction mean, so no shock boost is applied to them.
    """
    mass_factor = abs(rqm)

    n_norm = np.abs(moments.moment(phase_space, axis="p1", order=0))
    u_norm = moments.moment(phase_space, axis="p1", order=1) - v_shock
    vth2_sum = np.maximum(moments.moment(phase_space, axis="p1", order=2), 0.0)

    for perp in perp_phase_spaces or ():
        axis = _momentum_axis_name(perp)
        vth2_sum = vth2_sum + np.maximum(moments.moment(perp, axis=axis, order=2), 0.0)

    u_ram = 0.5 * n_norm * u_norm**2 * mass_factor
    u_th = 0.5 * n_norm * vth2_sum * mass_factor
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
        "full" uses the total B² — this is the real magnetic energy density and
        the correct choice for an energy budget / conservation check.
        "delta" subtracts the upstream mean B first, so it reports only the
        energy in the shock-*generated* field perturbation (compression +
        waves), discarding the pre-existing background.  Use "delta" only when
        you specifically want the perturbation energy, not for budgets.

        Note: "delta" is NOT a reference-frame transformation.  Boosting to the
        shock frame mixes E and B (E' = E + (v/c)×B in Gaussian units), but for
        this non-relativistic shock (v_sh ~ 0.04 c) those corrections are
        O(v²/c²) ~ 1e-3 and are neglected here; the lab-frame fields are used
        directly in both modes.
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
