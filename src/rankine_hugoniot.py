"""Rankine--Hugoniot (MHD) jump-condition baseline for collisionless shocks.

The point of this module is to answer "how much of the measured downstream
heating / compression is *just* the ideal-MHD shock jump?".  An ideal-MHD shock
with the same upstream state, shock speed, and field obliquity has a definite
downstream compression and temperature; the *excess* of the kinetically measured
downstream temperature over that prediction is the **anomalous (collisionless)
heating**, and the measured compression ratio can be checked against the theory.

This is the **oblique** MHD shock, parameterised by the upstream sonic and
Alfvenic Mach numbers and the shock-normal angle ``theta`` = theta_Bn.  The
perpendicular (theta = 90 deg) and parallel/hydrodynamic limits both fall out of
the same solver and are used as test anchors.

How the solver is built (so each step is checkable against a textbook):
the compression ratio ``X = rho2/rho1 = u1/u2x`` is found by root-finding the
**energy** jump condition, after the *lower-order* conservation laws are used to
express every other downstream quantity as a function of X:

    mass + induction + tangential momentum  ->  Bt2/Bt1(X)  and  u2y(X)
    normal momentum                         ->  P2(X)
    energy                                  ->  residual(X) = 0   (solve for X)

All speeds are in units of the upstream normal inflow speed ``u1`` and densities
in units of ``rho1`` while solving, so the core is **unit-agnostic**: it takes
Mach numbers and works unchanged in OSIRIS-normalised units, CGS, or SI.  The
convenience wrapper :func:`solve_jump` builds the Mach numbers from upstream
primitives in the same convention as ``scripts/dimensionless_params.py``.

Adiabatic index ``gamma``
-------------------------
``gamma`` is an explicit argument everywhere (module default ``GAMMA_DEFAULT =
5/3``); nothing hard-codes it inside a formula.  For ``f`` active velocity
degrees of freedom ``gamma = (f + 2) / f``:

    f = 3  ->  gamma = 5/3   (full 3D thermal motion)
    f = 2  ->  gamma = 2     (often the relevant value for a 1D PIC run)
    f = 1  ->  gamma = 3

so the measured compression can be compared against the prediction over a small
sweep of ``gamma`` (see :func:`gamma_from_dof`) to read off the effective index.

Sign / geometry conventions
----------------------------
The shock normal is x1; ``b1`` is the normal field component, ``b2``/``b3`` the
transverse ones.  ``theta`` = 0 is a parallel shock, ``theta`` = pi/2 a
perpendicular shock.  The upstream field magnitude sets the Alfven speed and
hence ``mach_a`` (total field, not just the normal component).
"""

import numpy as np
import scipy.optimize

GAMMA_DEFAULT = 5.0 / 3.0


def gamma_from_dof(f: int) -> float:
    """Adiabatic index for ``f`` active (velocity) degrees of freedom: (f+2)/f.

    f = 3 -> 5/3,  f = 2 -> 2,  f = 1 -> 3.  Handy for sweeping ``gamma`` to find
    the effective index of a reduced-dimensionality PIC run.
    """
    return (f + 2.0) / f


# ---------------------------------------------------------------------------
# Geometry: shock-normal angle and classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Oblique MHD shock adiabatic: compression ratio from Mach numbers + angle
# ---------------------------------------------------------------------------

def _downstream_from_X(X, inv_Ms2, inv_Ma2, cos2, sin2, gamma):
    """Downstream tangential-field ratio, tangential velocity, and pressure at X.

    Built from the conservation laws *below* the energy equation, in units where
    the upstream density and normal inflow speed are 1:

      * ``b = Bt2/Bt1``  from mass + induction + tangential momentum:
            Bt2/Bt1 = X (1 - cos2/M_A^2) / (1 - X cos2/M_A^2)
        (reduces to ``X`` for a perpendicular shock, cos2 = 0).
      * ``u2y`` (downstream tangential flow speed) from tangential momentum:
            u2y = (b - 1) sin(th)cos(th) / M_A^2
      * ``P2`` (downstream pressure, units rho1 u1^2) from normal momentum,
        with magnetic pressure from the *tangential* field only (the normal
        field is continuous and cancels):
            P2 = 1 - 1/X + 1/(gamma M_s^2) + (sin2 / (2 M_A^2)) (1 - b^2)
    """
    a = cos2 * inv_Ma2                       # = (v_Ax / u1)^2, normal Alfven term
    b = X * (1.0 - a) / (1.0 - X * a)        # Bt2 / Bt1
    sc = np.sqrt(sin2 * cos2)                # sin(theta) cos(theta) >= 0
    u2y = (b - 1.0) * sc * inv_Ma2           # downstream tangential flow speed
    P2 = (1.0 - 1.0 / X) + inv_Ms2 / gamma + 0.5 * sin2 * inv_Ma2 * (1.0 - b * b)
    return b, u2y, P2


def _energy_residual(X, inv_Ms2, inv_Ma2, cos2, sin2, gamma):
    """Energy-flux jump residual F_E2(X) - F_E1 (units rho1 u1^3); zero at the shock.

    The energy flux is  u_x (1/2 rho u^2 + gamma/(gamma-1) P) + Poynting_x, with
    Poynting_x = (u_x Bt^2 - u_y Bn Bt) / 4pi in Gaussian units.  Upstream is
    purely normal flow (u1y = 0); downstream carries u2y and a compressed Bt.
    """
    b, u2y, P2 = _downstream_from_X(X, inv_Ms2, inv_Ma2, cos2, sin2, gamma)

    # Upstream (u1 = rho1 = 1, u1y = 0, Bt1^2/4pi = sin2 / M_A^2):
    F_up = 0.5 + inv_Ms2 / (gamma - 1.0) + sin2 * inv_Ma2

    # Downstream (rho2 = X, u2x = 1/X, Bt2 = b Bt1):
    u2_sq = 1.0 / X**2 + u2y**2
    enthalpy = gamma / ((gamma - 1.0) * X) * P2
    poynting = b * b * sin2 * inv_Ma2 / X \
        - b * (b - 1.0) * sin2 * cos2 * inv_Ma2**2
    F_dn = 0.5 * u2_sq + enthalpy + poynting

    return F_dn - F_up


def compression_ratio(mach_s: float, mach_a: float, theta: float = np.pi / 2.0,
                      gamma: float = GAMMA_DEFAULT) -> float:
    """Density compression ratio r = rho2/rho1 for an oblique MHD shock.

    Solves the ideal-MHD shock adiabatic (mass, normal+tangential momentum,
    induction, and energy incl. Poynting flux) for the compressive (fast-shock)
    root r in (1, (gamma+1)/(gamma-1)].

    Parameters
    ----------
    mach_s : float
        Upstream sonic Mach number u1 / c_s.
    mach_a : float
        Upstream Alfvenic Mach number u1 / v_A, using the **total** upstream
        field.  Pass ``np.inf`` for the unmagnetised (hydrodynamic) limit.
    theta : float
        Shock-normal angle theta_Bn [rad].  pi/2 = perpendicular, 0 = parallel.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Compression ratio r >= 1, or ``nan`` if there is no compressive
        (super-fast) shock solution in the bracket.
    """
    if not (np.isfinite(mach_s) and mach_s > 0.0):
        return float("nan")

    inv_Ms2 = 1.0 / mach_s**2
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a**2
    cos2 = float(np.cos(theta))**2
    sin2 = float(np.sin(theta))**2

    args = (inv_Ms2, inv_Ma2, cos2, sin2, gamma)

    r_max = (gamma + 1.0) / (gamma - 1.0)
    lo = 1.0 + 1e-9
    hi = r_max
    # Bt2/Bt1 diverges at the intermediate (Alfven) point X = M_A^2 / cos^2;
    # the fast-shock root sits below it, so keep the bracket strictly under it.
    a = cos2 * inv_Ma2
    if a > 0.0:
        hi = min(hi, (1.0 / a) * (1.0 - 1e-9))
    if hi <= lo:
        return float("nan")

    f_lo = _energy_residual(lo, *args)
    f_hi = _energy_residual(hi, *args)
    if f_lo * f_hi > 0.0:
        return float("nan")     # no sign change => no compressive shock here
    return float(scipy.optimize.brentq(
        _energy_residual, lo, hi, args=args, xtol=1e-12, rtol=1e-12))


def perp_compression_ratio(mach_s: float, mach_a: float,
                           gamma: float = GAMMA_DEFAULT) -> float:
    """Perpendicular (theta = 90 deg) compression ratio; thin :func:`compression_ratio`."""
    return compression_ratio(mach_s, mach_a, theta=np.pi / 2.0, gamma=gamma)


def tangential_field_ratio(r: float, mach_a: float, theta: float) -> float:
    """Tangential field jump Bt2/Bt1 for an oblique MHD shock.

        Bt2/Bt1 = r (M_A^2 - cos^2 theta) / (M_A^2 - r cos^2 theta)

    This is *not* equal to the density compression r except for a perpendicular
    shock (cos theta = 0, where it collapses to r).  It is the relation that
    reconciles the two empirical readings of the compression: the density ratio
    n2/n1 = r and the transverse-field ratio |Bt2|/|Bt1|, given the angle.
    """
    if not np.isfinite(r):
        return float("nan")
    cos2 = float(np.cos(theta))**2
    if not np.isfinite(mach_a):
        return float(r)                      # unmagnetised: passive field, ~r
    Ma2 = mach_a**2
    denom = Ma2 - r * cos2
    if denom == 0.0:
        return float("inf")
    return float(r * (Ma2 - cos2) / denom)


# ---------------------------------------------------------------------------
# Downstream state from upstream primitives (OSIRIS units)
# ---------------------------------------------------------------------------

def solve_jump(
    n_e1: float,
    T_e1: float,
    T_i1: float,
    B2_1: float,
    abs_rqm_i: float,
    v_inflow: float,
    theta: float = np.pi / 2.0,
    gamma: float = GAMMA_DEFAULT,
) -> dict:
    """Oblique-MHD downstream state from upstream primitives (OSIRIS units).

    Mirrors the upstream-quantity conventions of
    ``scripts/dimensionless_params.py``::

        c_s = sqrt( gamma * (T_e + T_i) / |rqm_i| )      [c]
        v_A = sqrt( B^2 / (|rqm_i| * n_e) )              [c]   (total field)

    Parameters
    ----------
    n_e1, T_e1, T_i1 : float
        Upstream electron density [n_0] and electron/ion temperatures [m_e c^2].
    B2_1 : float
        Upstream B^2 = b1^2 + b2^2 + b3^2 [B_0^2] (total field, for v_A).
    abs_rqm_i : float
        |rqm_i| = m_i/(Z_i m_e), the OSIRIS ion mass-per-charge [m_e/e].
    v_inflow : float
        Shock-frame normal inflow speed [c] (= v_shock for upstream at rest).
    theta : float
        Shock-normal angle theta_Bn [rad]; pi/2 = perpendicular (default).
    gamma : float
        Adiabatic index.

    Returns
    -------
    dict
        ``r`` (density compression), ``B_ratio`` (tangential field jump
        Bt2/Bt1), ``p_ratio`` (p2/p1), ``T_factor`` (T2/T1 = p_ratio / r),
        ``mach_s``, ``mach_a``, ``theta``, and ``T_adiabatic`` (predicted
        downstream total temperature T_e+T_i in [m_e c^2]).
    """
    c_s2 = gamma * (T_e1 + T_i1) / abs_rqm_i        # sound speed^2 [c^2]
    v_A2 = B2_1 / (abs_rqm_i * n_e1)                # Alfven speed^2 [c^2] (total B)

    mach_s = v_inflow / np.sqrt(c_s2) if c_s2 > 0.0 else float("inf")
    mach_a = v_inflow / np.sqrt(v_A2) if v_A2 > 0.0 else float("inf")

    r = compression_ratio(mach_s, mach_a, theta=theta, gamma=gamma)
    if not np.isfinite(r):
        return {
            "r": float("nan"), "B_ratio": float("nan"), "p_ratio": float("nan"),
            "T_factor": float("nan"), "mach_s": mach_s, "mach_a": mach_a,
            "theta": theta, "T_adiabatic": float("nan"),
        }

    b = tangential_field_ratio(r, mach_a, theta)

    # Pressure jump (oblique), reusing the same dimensionless groups as the core:
    #   p2/p1 = 1 + gamma M_s^2 (1 - 1/r)
    #             + (gamma M_s^2 sin^2 / (2 M_A^2)) (1 - b^2)
    # (collapses to the verified perpendicular formula at theta = 90, b = r).
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a**2
    sin2 = float(np.sin(theta))**2
    p_ratio = 1.0 + gamma * mach_s**2 * (1.0 - 1.0 / r) \
        + 0.5 * gamma * mach_s**2 * sin2 * inv_Ma2 * (1.0 - b * b)
    T_factor = p_ratio / r                          # T ~ p / n

    return {
        "r": r,
        "B_ratio": b,                               # Bt2/Bt1 (oblique)
        "p_ratio": p_ratio,
        "T_factor": T_factor,
        "mach_s": mach_s,
        "mach_a": mach_a,
        "theta": theta,
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
