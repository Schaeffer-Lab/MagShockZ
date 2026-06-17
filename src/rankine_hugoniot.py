"""Rankine--Hugoniot (MHD) jump-condition baseline for collisionless shocks.

The point of this module is to answer "how much of the measured downstream
heating is *just* adiabatic compression?".  An ideal-MHD shock with the same
upstream state and shock speed has a definite downstream temperature; the
*excess* of the kinetically measured downstream temperature over that
prediction is the **anomalous (collisionless) heating**.

Everything here is pure (numpy/scipy only) and unit-agnostic: the core solver
takes the upstream **sonic** and **Alfvenic** Mach numbers, so it works
unchanged in OSIRIS-normalised units, CGS, or SI.  The convenience wrappers
build those Mach numbers from upstream primitives in the same convention as
``scripts/compute_dimensionless_params.py`` (OSIRIS units: T in m_e c^2, B^2 in
B_0^2, n in n_0, velocities in c, ion mass-per-charge ``abs_rqm_i`` in m_e/e).

Primary case is the **perpendicular** MHD shock (B transverse to the shock
normal), appropriate for this magnetized/quasi-perpendicular campaign.  The
hydrodynamic limit (M_A -> infinity) falls out automatically and is used as a
test anchor (strong-shock compression r -> (gamma+1)/(gamma-1) = 4).
"""

import numpy as np
import scipy.optimize

GAMMA_DEFAULT = 5.0 / 3.0


def shock_normal_angle(b1, b2, b3) -> float:
    """Angle theta_Bn [rad] between B and the shock normal (the x1 axis).

    In the 1D runs the shock normal is x1, so ``b1`` is the normal component and
    ``b2``/``b3`` are transverse.  Accepts scalars or arrays (arrays are mean-
    reduced, e.g. an upstream window).  theta_Bn = 0 is a parallel shock,
    pi/2 a perpendicular shock.
    """
    bn = float(np.nanmean(b1))
    bt = float(np.hypot(np.nanmean(b2), np.nanmean(b3)))
    return float(np.arctan2(abs(bt), abs(bn)))


def is_quasi_perpendicular(theta_bn: float) -> bool:
    """True if the shock is quasi-perpendicular (theta_Bn > 45 deg)."""
    return theta_bn > (np.pi / 4.0)


def perp_compression_ratio(mach_s: float, mach_a: float, gamma: float = GAMMA_DEFAULT) -> float:
    """Density compression ratio r = rho2/rho1 for a perpendicular MHD shock.

    Solves the ideal-MHD jump conditions (mass, momentum incl. magnetic
    pressure, energy incl. Poynting flux, induction B/rho = const) for the
    non-trivial root r in (1, (gamma+1)/(gamma-1)].

    Parameters
    ----------
    mach_s : float
        Upstream sonic Mach number v_inflow / c_s.
    mach_a : float
        Upstream Alfvenic Mach number v_inflow / v_A.  Pass ``np.inf`` for the
        unmagnetised (hydrodynamic) limit.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Compression ratio r >= 1.  Returns ``nan`` if the flow is sub-critical
        (no compressive shock solution, i.e. fast Mach number <= 1).
    """
    if not (np.isfinite(mach_s) and mach_s > 0.0):
        return float("nan")

    inv_Ms2 = 1.0 / mach_s**2
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a**2

    # Energy condition (mass + momentum already substituted), divided by v1^2:
    #   1/2 + 1/((g-1) Ms^2) + 1/Ma^2
    #     = 1/(2 r^2)
    #       + 1/((g-1) r) [ 1/Ms^2 + g (1 - 1/r) - (g/2) (r^2-1)/Ma^2 ]
    #       + r / Ma^2
    lhs = 0.5 + inv_Ms2 / (gamma - 1.0) + inv_Ma2

    def residual(r):
        bracket = inv_Ms2 + gamma * (1.0 - 1.0 / r) - 0.5 * gamma * (r**2 - 1.0) * inv_Ma2
        rhs = 0.5 / r**2 + bracket / ((gamma - 1.0) * r) + r * inv_Ma2
        return rhs - lhs

    r_max = (gamma + 1.0) / (gamma - 1.0)
    # The trivial root sits at r = 1; bracket strictly above it.
    lo, hi = 1.0 + 1e-9, r_max
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        # No sign change => no compressive (super-fast) shock solution.
        return float("nan")
    return float(scipy.optimize.brentq(residual, lo, hi, xtol=1e-12, rtol=1e-12))


def solve_jump(
    n_e1: float,
    T_e1: float,
    T_i1: float,
    B2_1: float,
    abs_rqm_i: float,
    v_inflow: float,
    gamma: float = GAMMA_DEFAULT,
) -> dict:
    """Perpendicular-MHD downstream state from upstream primitives (OSIRIS units).

    Mirrors the upstream-quantity conventions of
    ``scripts/compute_dimensionless_params.py``::

        c_s = sqrt( gamma * (T_e + T_i) / |rqm_i| )      [c]
        v_A = sqrt( B^2 / (|rqm_i| * n_e) )              [c]

    Parameters
    ----------
    n_e1, T_e1, T_i1 : float
        Upstream electron density [n_0] and electron/ion temperatures [m_e c^2].
    B2_1 : float
        Upstream B^2 = b1^2 + b2^2 + b3^2 [B_0^2].
    abs_rqm_i : float
        |rqm_i| = m_i/(Z_i m_e), the OSIRIS ion mass-per-charge [m_e/e].
    v_inflow : float
        Shock-frame inflow speed [c] (= v_shock for upstream at rest).
    gamma : float
        Adiabatic index.

    Returns
    -------
    dict
        ``r`` (compression), ``B_ratio`` (= r for a perpendicular shock),
        ``p_ratio`` (p2/p1), ``T_factor`` (T2/T1 = p_ratio / r),
        ``mach_s``, ``mach_a``, and ``T_adiabatic`` (predicted downstream
        total temperature T_e+T_i in [m_e c^2]).
    """
    p1 = n_e1 * (T_e1 + T_i1)                       # total thermal pressure [n_0 m_e c^2]
    c_s2 = gamma * (T_e1 + T_i1) / abs_rqm_i        # sound speed^2 [c^2]
    v_A2 = B2_1 / (abs_rqm_i * n_e1)                # Alfven speed^2 [c^2]

    mach_s = v_inflow / np.sqrt(c_s2) if c_s2 > 0.0 else float("inf")
    mach_a = v_inflow / np.sqrt(v_A2) if v_A2 > 0.0 else float("inf")

    r = perp_compression_ratio(mach_s, mach_a, gamma)
    if not np.isfinite(r):
        return {
            "r": float("nan"), "B_ratio": float("nan"), "p_ratio": float("nan"),
            "T_factor": float("nan"), "mach_s": mach_s, "mach_a": mach_a,
            "T_adiabatic": float("nan"),
        }

    # Momentum jump (perpendicular): p2/p1 = 1 + g Ms^2 (1 - 1/r)
    #                                          - (g/2)(Ms^2/Ma^2)(r^2 - 1)
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a**2
    p_ratio = 1.0 + gamma * mach_s**2 * (1.0 - 1.0 / r) \
        - 0.5 * gamma * mach_s**2 * inv_Ma2 * (r**2 - 1.0)
    T_factor = p_ratio / r                          # T ~ p / n

    return {
        "r": r,
        "B_ratio": r,                               # B2/B1 = r (perpendicular)
        "p_ratio": p_ratio,
        "T_factor": T_factor,
        "mach_s": mach_s,
        "mach_a": mach_a,
        "T_adiabatic": (T_e1 + T_i1) * T_factor,
    }


def anomalous_heating(T_measured_dn: float, T_upstream: float, T_factor: float) -> dict:
    """Split measured downstream heating into adiabatic and anomalous parts.

    The MHD jump heats the plasma by ``T_factor`` (same factor applied to a
    species temperature under single-fluid adiabatic compression).  The excess
    of the kinetically measured downstream temperature over that is the
    collisionless/anomalous contribution.

    Parameters
    ----------
    T_measured_dn : float
        Kinetically measured downstream temperature (per species or total).
    T_upstream : float
        Matching upstream temperature, same units/quantity.
    T_factor : float
        Adiabatic temperature ratio T2/T1 from :func:`solve_jump`.

    Returns
    -------
    dict
        ``adiabatic`` (predicted downstream T), ``anomalous`` (measured -
        predicted), ``total_heating`` (measured - upstream), and
        ``anomalous_frac`` (anomalous / total_heating).
    """
    adiabatic = T_upstream * T_factor
    anomalous = T_measured_dn - adiabatic
    total_heating = T_measured_dn - T_upstream
    frac = anomalous / total_heating if total_heating != 0.0 else float("nan")
    return {
        "adiabatic": adiabatic,
        "anomalous": anomalous,
        "total_heating": total_heating,
        "anomalous_frac": frac,
    }
