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


def species_momentum_fluxes(
    phase_space,
    rqm: float,
    v_shock: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_ram, p_th) shock-normal momentum-flux channels for one species.

    These are the **pressure** companions of :func:`species_energy_profiles`: the
    conserved Rankine--Hugoniot quantity across a steady shock is the normal
    momentum flux ``Π_xx = ρ U² + P_xx + B_t²/2`` (continuous up/downstream),
    not the energy density.  In OSIRIS units (``n_0 m_e c²``, ``u = v/c``):

        p_ram = n |rqm| U²        ram (dynamic) pressure   [= 2 · u_ram]
        p_th  = n |rqm| σ_p1²     shock-normal pressure P_xx

    Only the shock-normal direction enters (unlike the energy thermal channel,
    which sums every momentum direction): the *normal* momentum flux carries the
    normal bulk velocity ``U = ⟨u_p1⟩ − v_shock`` and the normal pressure
    ``P_xx``.  Transverse phase spaces are therefore not used here (they feed the
    *tangential* momentum balance, not this normal one).  For an isotropic
    plasma ``P_xx = (2/3) u_th``.

    Parameters mirror :func:`species_energy_profiles` (sans ``perp_phase_spaces``).
    """
    mass_factor = abs(rqm)

    n_norm = np.abs(moments.moment(phase_space, axis="p1", order=0))
    u_norm = moments.moment(phase_space, axis="p1", order=1) - v_shock
    sx2 = np.maximum(moments.moment(phase_space, axis="p1", order=2), 0.0)

    p_ram = n_norm * u_norm**2 * mass_factor
    p_th = n_norm * sx2 * mass_factor
    return p_ram, p_th


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


def transverse_magnetic_pressure(b2: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """Transverse magnetic pressure ½(b2² + b3²) in OSIRIS units (n_0 m_e c²).

    The magnetic contribution to the *normal* momentum flux is the transverse
    field pressure B_t²/8π → ½(b2² + b3²) here (x1 is the shock normal, so b1 is
    the normal field, which is continuous and carries magnetic *tension* — not
    normal momentum flux — and is excluded).  The electric momentum flux
    (Maxwell stress ∝ E²) is negligible for this non-relativistic shock and is
    dropped, matching the MHD jump condition ρU² + p + B_t²/8π.
    """
    b2, b3 = np.asarray(b2), np.asarray(b3)
    return 0.5 * (np.square(b2) + np.square(b3))


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


def momentum_partition_by_region(
    channels: dict,
    x_axis: np.ndarray,
    x_shock: float,
    x_downstream_start: float,
) -> dict:
    """Average an arbitrary set of momentum-flux channels over each region.

    Generic companion to :func:`partition_by_region` for the pressure channels
    (:func:`species_momentum_fluxes`, :func:`transverse_magnetic_pressure`).
    Pass a dict ``{name: profile}`` of simulation-unit arrays on a common grid.

    Returns
    -------
    dict with "upstream"/"downstream", each ``{means, fractions, total}`` (so
    the total is the conserved momentum flux — feed it to :func:`continuity_check`).
    """
    upstream, downstream = region_masks(x_axis, x_shock, x_downstream_start)

    if not downstream.any() or not upstream.any():
        raise ValueError(
            f"Empty region mask — check x_shock={x_shock:.1f} and "
            f"x_downstream_start={x_downstream_start:.1f} vs "
            f"x_axis=[{x_axis.min():.1f}, {x_axis.max():.1f}]"
        )

    def _side(mask):
        means = {k: float(np.nanmean(v[mask])) for k, v in channels.items()}
        total = sum(means.values())
        fracs = {k: (v / total if total else float("nan")) for k, v in means.items()}
        return {"means": means, "fractions": fracs, "total": total}

    return {"upstream": _side(upstream), "downstream": _side(downstream)}


def continuity_check(result: dict) -> dict:
    """Downstream/upstream ratio of the total (and each channel) momentum flux.

    For the conserved momentum flux (:func:`momentum_partition_by_region`) the
    total ``dn/up`` ≈ 1 across a steady shock — the quantitative continuity test.

    Returns ``{total_up, total_dn, ratio, rel_imbalance, channels}`` where
    ``channels`` is the per-channel dn/up dict.
    """
    up = result["upstream"]["total"]
    dn = result["downstream"]["total"]
    channels = {
        ch: (result["downstream"]["means"][ch] / result["upstream"]["means"][ch]
             if result["upstream"]["means"][ch] else float("nan"))
        for ch in result["upstream"]["means"]
    }
    return {
        "total_up": up,
        "total_dn": dn,
        "ratio": dn / up if up else float("nan"),
        "rel_imbalance": (dn - up) / up if up else float("nan"),
        "channels": channels,
    }


def continuity_summary(check: dict, unit: str = "n₀ mₑ c²") -> str:
    """Return a formatted continuity-check table from :func:`continuity_check`."""
    sep = "-" * 56
    return "\n".join([
        sep,
        "  Momentum-flux continuity  (conserved if dn/up ≈ 1)",
        sep,
        f"  total upstream   = {check['total_up']:.3e} {unit}",
        f"  total downstream = {check['total_dn']:.3e} {unit}",
        f"  dn/up = {check['ratio']:.3f}   ({100 * check['rel_imbalance']:+.1f}%)",
        sep,
    ])
