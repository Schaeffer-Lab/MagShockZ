"""perpendicular_shock.py -- compression ratio of a PERPENDICULAR MHD shock.

A deliberately simple, self-contained calculation of the theta = 90 degrees
(perpendicular) ideal-MHD shock jump, meant to be read top to bottom and checked
against the textbook line by line.  It follows Fitzpatrick's *Plasma Physics*
lecture notes, section "Perpendicular shocks":

    https://farside.ph.utexas.edu/teaching/plasma/Plasma/node104.html

The equation numbers below ((7.273), (7.274), (7.276)) are that page's.

What "perpendicular MHD shock" assumes
--------------------------------------
* Ideal MHD: the plasma is a single, perfectly-conducting fluid with a SCALAR
  (isotropic) pressure p.  There is no separate electron/ion temperature, no
  pressure anisotropy, no kinetic (collisionless) physics -- those are exactly
  the effects a PIC code like OSIRIS keeps and this theory drops.  This file is
  the simplest possible baseline; the kinetic corrections are a separate, much
  more involved calculation.
* Perpendicular geometry: the upstream flow V1 is along the shock normal (x) and
  the magnetic field B1 is purely transverse (y).  Then the field is simply
  compressed with the density, B2/B1 = rho2/rho1 = r, and the downstream flow has
  no transverse component.

Geometry / notation (upstream = 1, downstream = 2; shock at rest):

    V1 = (V1, 0, 0)      B1 = (0, B1, 0)
    V2 = (V2, 0, 0)      B2 = (0, B2, 0)

    r   = rho2/rho1 = B2/B1 = V1/V2     (the compression ratio we solve for)

Everything reduces to TWO dimensionless upstream numbers, so it works unchanged
whatever unit system the data came in (OSIRIS-normalised, CGS, SI):

    sonic Mach number     M_s = V1 / c_s      c_s = sqrt(gamma p1 / rho1)
    Alfvenic Mach number  M_A = V1 / v_A      v_A = B1 / sqrt(mu0 rho1)   [SI]
                                                  = B1 / sqrt(4 pi rho1)  [Gaussian]

The plasma beta used by the textbook is the ratio of thermal to magnetic
pressure, which in terms of those Mach numbers is (see :func:`plasma_beta`):

    beta1 = 2 mu0 p1 / B1^2 = (2 / gamma) (M_A / M_s)^2

Plugging in data
----------------
The lowest-level entry point is :func:`solve` (give it M_s and M_A directly).
If you have speeds, call :func:`solve_from_speeds` with V1, c_s, v_A in ONE
consistent unit system.  If you have the raw plasma state (densities, two
temperatures, B, rho, v_shock) -- as FLASH/OSIRIS region averages do -- hand it
straight to :func:`solve_from_upstream`, which forms c_s (:func:`sound_speed`,
two-temperature) and v_A (:func:`alfven_speed`) for you.  All of these are
unit-agnostic: pass unyt quantities (units travel through) or bare CGS floats.
"""

import numpy as np

GAMMA_DEFAULT = 5.0 / 3.0


def plasma_beta(mach_s: float, mach_a: float, gamma: float = GAMMA_DEFAULT) -> float:
    """Upstream plasma beta from the two Mach numbers.

    beta1 = (thermal pressure) / (magnetic pressure) = 2 mu0 p1 / B1^2.

    Using c_s^2 = gamma p1 / rho1 and v_A^2 = B1^2 / (mu0 rho1),

        beta1 = 2 p1 / (B1^2 / mu0) = (2 / gamma) (c_s / v_A)^2
              = (2 / gamma) (M_A / M_s)^2

    because c_s/v_A = (V1/M_s)/(V1/M_A) = M_A/M_s.
    """
    return (2.0 / gamma) * (mach_a / mach_s) ** 2


def shock_exists(mach_s: float, mach_a: float, gamma: float = GAMMA_DEFAULT) -> bool:
    """True if a compressive perpendicular shock can form.

    A perpendicular shock requires the inflow to be super-fast-magnetosonic,
    V1^2 > c_s^2 + v_A^2 (Eq. 7.277), equivalently (Eq. 7.276)

        M_s^2 > 1 + 2 / (gamma beta1).

    In Mach-number form V1^2 > c_s^2 + v_A^2 is just 1 > 1/M_s^2 + 1/M_A^2.
    """
    if not (np.isfinite(mach_s) and mach_s > 0.0):
        return False
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a ** 2
    return (1.0 / mach_s ** 2 + inv_Ma2) < 1.0


def compression_ratio(mach_s: float, mach_a: float,
                      gamma: float = GAMMA_DEFAULT) -> float:
    """Density compression ratio r = rho2/rho1 for a perpendicular MHD shock.

    Solves the quadratic (Fitzpatrick Eq. 7.274)

        F(r) = 2 (2 - gamma) r^2
             + gamma [ 2 (1 + beta1) + (gamma - 1) beta1 M_s^2 ] r
             - gamma (gamma + 1) beta1 M_s^2
             = 0

    and returns the physical (compressive) root, which lies in the range
    1 < r <= (gamma + 1)/(gamma - 1).  Returns ``nan`` if no compressive shock
    exists (see :func:`shock_exists`).

    Pass ``mach_a = np.inf`` for the unmagnetised limit; then beta1 -> inf and the
    quadratic reduces to the ordinary gas-dynamic shock
    r = (gamma + 1) M_s^2 / ((gamma - 1) M_s^2 + 2).
    """
    if not shock_exists(mach_s, mach_a, gamma):
        return float("nan")

    # Unmagnetised limit: beta1 -> inf, handle the gas-dynamic shock directly so
    # we do not divide by an infinite beta in the quadratic.
    if not np.isfinite(mach_a):
        M2 = mach_s ** 2
        return (gamma + 1.0) * M2 / ((gamma - 1.0) * M2 + 2.0)

    beta1 = plasma_beta(mach_s, mach_a, gamma)
    M2 = mach_s ** 2

    # Quadratic coefficients A r^2 + B r + C = 0  (Eq. 7.274).
    A = 2.0 * (2.0 - gamma)
    B = gamma * (2.0 * (1.0 + beta1) + (gamma - 1.0) * beta1 * M2)
    C = -gamma * (gamma + 1.0) * beta1 * M2

    r_max = (gamma + 1.0) / (gamma - 1.0)

    if A == 0.0:                       # gamma == 2 makes the equation linear
        roots = [-C / B] if B != 0.0 else []
    else:
        roots = np.roots([A, B, C]).tolist()

    # Keep the real, compressive root within the strong-shock ceiling.
    physical = [float(np.real(z)) for z in roots
                if abs(np.imag(z)) < 1e-9 and 1.0 < np.real(z) <= r_max + 1e-9]
    if not physical:
        return float("nan")
    return max(physical)


def pressure_ratio(r: float, mach_s: float, mach_a: float,
                   gamma: float = GAMMA_DEFAULT) -> float:
    """Downstream/upstream pressure ratio R = p2/p1 (Fitzpatrick Eq. 7.273).

        R = 1 + gamma M_s^2 (1 - 1/r) + (1/beta1) (1 - r^2)

    The middle term is the gas-dynamic pressure rise; the last term is the work
    done against the compressed magnetic field (it lowers R relative to the pure
    hydro shock).
    """
    if not np.isfinite(r):
        return float("nan")
    inv_beta = 0.0 if not np.isfinite(mach_a) else 1.0 / plasma_beta(mach_s, mach_a, gamma)
    return 1.0 + gamma * mach_s ** 2 * (1.0 - 1.0 / r) + inv_beta * (1.0 - r ** 2)


def solve(mach_s: float, mach_a: float, gamma: float = GAMMA_DEFAULT) -> dict:
    """Full perpendicular-shock jump from the two upstream Mach numbers.

    Returns a dict with the compression ratio and the dependent ratios; every
    field is dimensionless so it is independent of the input unit system:

        r          : rho2/rho1 = B2/B1 = V1/V2   (compression)
        p_ratio    : p2/p1                        (Eq. 7.273)
        T_ratio    : T2/T1 = (p2/p1)/(rho2/rho1) = p_ratio / r
        beta1      : upstream plasma beta
        mach_s     : sonic Mach number (echoed back)
        mach_a     : Alfvenic Mach number (echoed back)
        exists     : whether a compressive shock forms
    """
    exists = shock_exists(mach_s, mach_a, gamma)
    r = compression_ratio(mach_s, mach_a, gamma)
    p_ratio = pressure_ratio(r, mach_s, mach_a, gamma)
    return {
        "r": r,
        "p_ratio": p_ratio,
        "T_ratio": p_ratio / r if np.isfinite(r) and r > 0 else float("nan"),
        "beta1": plasma_beta(mach_s, mach_a, gamma) if np.isfinite(mach_a) else float("inf"),
        "mach_s": mach_s,
        "mach_a": mach_a,
        "exists": exists,
    }


def solve_from_speeds(v_inflow: float, c_s: float, v_A: float,
                      gamma: float = GAMMA_DEFAULT) -> dict:
    """Same as :func:`solve`, but plug in speeds straight from simulation data.

    ``v_inflow`` (the shock-frame inflow speed V1), ``c_s`` (sound speed) and
    ``v_A`` (Alfven speed) must all be in the SAME units (e.g. all in c for
    OSIRIS, all in cm/s for FLASH); only their ratios matter.  Pass ``v_A = 0``
    for the unmagnetised limit.
    """
    mach_s = v_inflow / c_s if c_s > 0.0 else float("inf")
    mach_a = v_inflow / v_A if v_A > 0.0 else float("inf")
    out = solve(mach_s, mach_a, gamma)
    out["c_s"] = c_s
    out["v_A"] = v_A
    out["v_inflow"] = v_inflow
    return out


# ---------------------------------------------------------------------------
# Upstream speeds from the raw plasma state (so callers hand over fields, not
# pre-computed speeds).  These do pure formula arithmetic, so unyt quantities
# pass through with their units and bare Gaussian-CGS floats work unchanged.
# ---------------------------------------------------------------------------

def sound_speed(ne, Te, n_ion, Ti, rho, *,
                gamma_e: float = GAMMA_DEFAULT, gamma_i: float = GAMMA_DEFAULT):
    """Sound speed of a two-temperature (T_e != T_i) plasma.

    Written in the general two-temperature (ion-acoustic) form,

        c_s = sqrt( (gamma_e Z k T_e + gamma_i k T_i) / M )

    with Z the charge state and M the ion mass (e.g.
    https://en.wikipedia.org/wiki/Ion_acoustic_wave).  We never need Z, M or
    k_B: multiply top and bottom by the ion density and use quasineutrality
    n_e = Z n_ion together with rho = M n_ion to get the same thing built only
    from measured quantities,

        c_s = sqrt( (gamma_e n_e k T_e + gamma_i n_ion k T_i) / rho )
            = sqrt( (gamma_e P_e + gamma_i P_i) / rho )

    since each n*kT is a partial pressure (T_e, T_i are passed as energies kT).

    Relation to the single-fluid MHD sound speed
    --------------------------------------------
    With gamma_e = gamma_i = gamma this is exactly c_s = sqrt(gamma (P_e+P_i)/rho)
    = sqrt(gamma P / rho), the single scalar-pressure, single-gamma sound speed
    that the perpendicular-shock jump assumes -- so the default keeps M_s
    consistent with the jump solver.  Decouple the indices (isothermal electrons
    gamma_e = 1, 1-D adiabatic ions gamma_i = 3, ...) only when you deliberately
    want the kinetic ion-acoustic speed instead of the MHD one.
    """
    return np.sqrt((gamma_e * ne * Te + gamma_i * n_ion * Ti) / rho)


def alfven_speed(B_perp, rho):
    """Alfven speed v_A = B / sqrt(4 pi rho) in Gaussian CGS.

    For a perpendicular shock the field is purely transverse, so ``B_perp`` is
    the field that enters v_A.  Unit-agnostic: a unyt B [Gauss] with rho
    [g/cm^3] returns cm/s automatically (Gauss = g^1/2 cm^-1/2 s^-1); bare
    floats must already be in Gaussian CGS.
    """
    return B_perp / np.sqrt(4.0 * np.pi * rho)


def mass_flux_shock_speed(rho_up, v_up, rho_dn, v_dn):
    """Shock speed from MASS-FLUX continuity across the front.

    In the shock frame the mass flux rho (v - v_sh) is continuous,
    rho_up (v_up - v_sh) = rho_dn (v_dn - v_sh).  Solving for the (lab-frame)
    shock speed,

        v_sh = (rho_dn v_dn - rho_up v_up) / (rho_dn - rho_up)

    where v_up, v_dn are the lab-frame bulk speeds along the shock normal.  This
    is the unique frame in which mass is conserved across the front; it follows
    from mass continuity ALONE (no momentum or energy closure), so it is the most
    robust shock-speed estimator and the right thing to track dump-by-dump for an
    accelerating shock.  Unit-agnostic (unyt quantities or consistent floats);
    returns nan when rho_dn == rho_up (no compression, speed undefined).
    """
    denom = rho_dn - rho_up
    if float(denom) == 0.0:
        return float("nan")
    return (rho_dn * v_dn - rho_up * v_up) / denom


def _ratio(a, b) -> float:
    """``a / b`` as a plain float (a, b same dimension; unyt or float)."""
    return float("inf") if float(b) == 0.0 else float(a / b)


def solve_from_upstream(*, ne, Te, n_ion, Ti, B_perp, rho, v_shock,
                        v_para=0.0, B_para=None,
                        gamma: float = GAMMA_DEFAULT,
                        gamma_e: float = None, gamma_i: float = None) -> dict:
    """Solve the perpendicular shock straight from the UPSTREAM plasma fields.

    Hand it the upstream (region-averaged) state and the shock speed; it builds
    the sound and Alfven speeds and the shock-frame inflow speed internally and
    solves the jump, so callers never assemble Mach numbers by hand.  All inputs
    must be in ONE consistent unit system -- unyt quantities (recommended; the
    returned speeds keep their units) or bare Gaussian-CGS floats.  Inputs:

        ne, n_ion : electron / ion number densities
        Te, Ti    : electron / ion temperatures as energies kT
        B_perp    : transverse (shock-tangential) field  -> v_A
        rho       : mass density
        v_shock   : shock speed (lab frame)
        v_para    : upstream bulk speed along the normal (default 0)
        B_para    : normal field component; if given, theta_bn is reported

    Internally:

        c_s      = sound_speed(...)              two-temperature, gamma_e/gamma_i
        v_A      = alfven_speed(B_perp, rho)     transverse field
        v_inflow = |v_shock - v_para|            shock-frame normal inflow
        theta_bn = atan2(|B_perp|, |B_para|)     obliquity (diagnostic only)

    Returns the :func:`solve` dict augmented with ``c_s``, ``v_A``,
    ``v_inflow`` (and ``theta_bn`` when ``B_para`` is given).  The jump itself
    is the theta = 90 deg perpendicular solution; theta_bn is reported only so
    you can see how perpendicular the data actually is.
    """
    if gamma_e is None:
        gamma_e = gamma
    if gamma_i is None:
        gamma_i = gamma

    c_s = sound_speed(ne, Te, n_ion, Ti, rho, gamma_e=gamma_e, gamma_i=gamma_i)
    v_A = alfven_speed(B_perp, rho)
    v_inflow = abs(v_shock - v_para)

    out = solve(_ratio(v_inflow, c_s), _ratio(v_inflow, v_A), gamma)
    out["c_s"] = c_s
    out["v_A"] = v_A
    out["v_inflow"] = v_inflow
    if B_para is not None:
        out["theta_bn"] = float(np.arctan2(abs(float(B_perp)), abs(float(B_para))))
    return out


def predict_downstream(jump: dict, *, rho1=None, p1=None, B_perp1=None,
                       T1=None, v_inflow=None) -> dict:
    """Map measured UPSTREAM values to predicted DOWNSTREAM values.

    Given a solved ``jump`` (from :func:`solve` / :func:`solve_from_speeds`) and
    whichever upstream quantities you measured, apply the perpendicular-shock
    jump relations.  Only the quantities you pass are predicted, and each keeps
    whatever units you gave it (the jump is pure ratios).  Downstream = 2,
    upstream = 1:

        rho2    = r * rho1                  mass continuity (compression)
        B_perp2 = r * B_perp1              frozen-in transverse field, B2/B1 = r
        p2      = (p2/p1) * p1             Eq. 7.273 (total scalar pressure)
        T2      = (T2/T1) * T1             T2/T1 = (p2/p1) / r
        v2      = v_inflow / r             shock-frame normal speed, V1/r

    Returns a dict carrying the predicted downstream values that were requested,
    plus ``r`` and ``p_ratio`` for reference.
    """
    r = jump["r"]
    out = {"r": r, "p_ratio": jump["p_ratio"], "T_ratio": jump["T_ratio"]}
    if rho1 is not None:
        out["rho"] = r * rho1
    if B_perp1 is not None:
        out["B_perp"] = r * B_perp1
    if p1 is not None:
        out["p"] = jump["p_ratio"] * p1
    if T1 is not None:
        out["T"] = jump["T_ratio"] * T1
    if v_inflow is not None:
        out["v_inflow"] = v_inflow / r
    return out
