"""Shock-frame energy-*flux* profiles for OSIRIS shock analysis.

Across a steady shock the conserved quantity is the energy **flux**, not the
energy **density** (the density jumps simply because the flow compresses and
slows).  The rigorous conservation check therefore compares the x-directed
energy flux upstream and downstream in the shock rest frame.

For a fluid the shock-normal energy flux is

    F = U_x ( ½ρ|U|² + ε + P_xx ) + q_x        (kinetic, per species)
        + (c/4π)(E × B)_x                       (electromagnetic / Poynting)

with U = v − v_shock the shock-frame velocity, ε = ½ρ⟨δv²⟩ the internal
(thermal) energy density and P_xx = ρ⟨δv_x²⟩ the shock-normal pressure.  The
three kinetic pieces are the advected bulk kinetic energy, the advected internal
energy, and the pressure work; their sum (for an isotropic ideal gas) is the
familiar enthalpy flux U(½ρU² + γ/(γ−1)·P).

Limitation: the collisionless heat flux q_x = ½ρ⟨δv_x|δv|²⟩ is a 3rd-order /
cross moment that the marginal pᵢx₁ phase spaces cannot provide, so it is
neglected.  The off-diagonal pressure work (P_xy u_y + P_xz u_z) is likewise not
available and dropped; the bulk transport keeps only U_x.

Units: everything is OSIRIS-normalised.  Velocities are in c, so the fluxes come
out in n_0 m_e c² — i.e. the physical flux divided by c — and thus share units
with the energy *densities* in ``energy_partition.py``.  In the Gaussian-based
OSIRIS normalization the lab-frame Poynting flux S_x/c reduces to E2·B3 − E3·B2.

Frame consistency (important): the kinetic channels use the shock-frame velocity
U = v − v_shock, so the electromagnetic flux must be evaluated in the *same*
frame to make ``F_up ≈ F_dn`` a valid steady-shock conservation check.  The
non-relativistic field transformation to a frame moving at +v_shock·x̂ is
E2' = E2 − (v_shock/c)·B3,  E3' = E3 + (v_shock/c)·B2 (Gaussian, E' = E +
(V/c)×B), which gives

    S_x'/c = E2'·B3 − E3'·B2 = (E2·B3 − E3·B2) − (v_shock/c)·(B2² + B3²).

The extra −(v_shock/c)·B_perp² term is the advection of the perpendicular
magnetic enthalpy (U·B²/4π in CGS) — a *leading-order* O(v_shock/c) term, the
field analog of the kinetic bulk/pressure advection, NOT an O(v²/c²) correction.
``poynting_flux`` takes ``v_shock`` and defaults to 0 (pure lab frame).
"""

import numpy as np

import moments


def _momentum_axis_name(phase_space) -> str:
    """Name of the momentum axis of a pNx1 phase space (the non-spatial axis)."""
    for ax in phase_space.axes:
        if ax.name.startswith("p"):
            return ax.name
    raise ValueError(
        "Phase space has no momentum axis (no axis whose name starts with 'p')"
    )


def species_energy_flux(phase_space, rqm: float, v_shock: float,
                        perp_phase_spaces=None):
    """Shock-frame kinetic energy-flux channels for one species.

    Returns ``(F_bulk, F_internal, F_pressure)`` profiles in n_0 m_e c² (flux/c):

    - ``F_bulk``     = U·½n|rqm|·|U|²  — advected bulk kinetic energy.  The
      perpendicular bulk velocities (from ``perp_phase_spaces``) enter |U|²; the
      transport velocity is the shock-normal U = ⟨u_p1⟩ − v_shock.
    - ``F_internal`` = U·ε,  ε = ½n|rqm|·Σ_d σ_d²  — advected internal energy
      (sums every available momentum direction, like ``species_energy_profiles``).
    - ``F_pressure`` = U·P_xx,  P_xx = n|rqm|·σ_p1²  — shock-normal pressure work.

    Parameters mirror ``energy_partition.species_energy_profiles``.
    """
    mass = abs(rqm)

    n = np.abs(moments.moment(phase_space, axis="p1", order=0))
    U = moments.moment(phase_space, axis="p1", order=1) - v_shock
    sx2 = np.maximum(moments.moment(phase_space, axis="p1", order=2), 0.0)

    var_sum = np.array(sx2, dtype=float, copy=True)   # Σ_d σ_d²
    u_perp2 = np.zeros_like(n, dtype=float)           # Σ_perp ⟨u⟩²
    for perp in perp_phase_spaces or ():
        axis = _momentum_axis_name(perp)
        var_sum = var_sum + np.maximum(moments.moment(perp, axis=axis, order=2), 0.0)
        u_perp2 = u_perp2 + moments.moment(perp, axis=axis, order=1) ** 2

    eps = 0.5 * n * mass * var_sum                    # internal energy density
    P_xx = n * mass * sx2                             # shock-normal pressure
    bulk_ke = 0.5 * n * mass * (U ** 2 + u_perp2)     # bulk KE density (shock frame)

    return U * bulk_ke, U * eps, U * P_xx


def poynting_flux(e2, e3, b2, b3, v_shock: float = 0.0):
    """Shock-normal Poynting flux S_x/c [n_0 m_e c²], in the shock frame.

    The x-component of (c/4π)(E×B) in OSIRIS Gaussian-based units, divided by c
    so it shares units with the kinetic fluxes and the energy densities, with the
    fields boosted to the frame moving at ``v_shock``·x̂ (see the module docstring
    for the derivation)::

        S_x'/c = (E2·B3 − E3·B2) − (v_shock/c)·(B2² + B3²)

    ``v_shock`` is in units of c.  ``v_shock = 0`` returns the lab-frame flux.
    The second term is the perpendicular magnetic-enthalpy advection and is the
    field analog of the kinetic advective channels in :func:`species_energy_flux`.
    """
    e2, e3, b2, b3 = map(np.asarray, (e2, e3, b2, b3))
    lab = e2 * b3 - e3 * b2
    return lab - v_shock * (b2**2 + b3**2)
