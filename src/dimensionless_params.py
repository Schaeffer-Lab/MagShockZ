"""Dimensionless plasma parameters for a magnetized collisionless shock.

Pure functions (numpy only) that turn region-averaged primitive quantities into
the dimensionless numbers that characterise the shock, all in OSIRIS normalised
units.  ``magnetic_reynolds`` is the one exception: it converts to physical units
via astropy/plasmapy (imported lazily, so this module stays CI-importable).

OSIRIS normalisation
--------------------
  lengths     c / omega_pe        velocities  c
  B fields    B_0 = m_e c omega_pe / e        densities  n_0        energies  m_e c^2

Crucial identity (Gaussian):  B_0^2 = 4 pi n_0 m_e c^2, hence in sim units
  magnetic pressure  P_B  = B_sim^2 / 2          [n_0 m_e c^2]
  thermal pressure   P_th = n_e (T_e + T_i)      [n_0 m_e c^2]

OSIRIS rqm = m/q (mass-per-charge) in units of m_e/e.  The charge state Z_i is
folded into rqm, so each sim ion macroparticle carries UNIT charge and mass
|rqm_i| m_e:  |rqm_i| = m_i / (Z_i m_e)  =>  physical mass ratio m_i/m_e = Z_i|rqm_i|.
Sim ions carry unit charge, so quasineutrality is simply n_i = n_e, and with
macroparticle mass |rqm_i| m_e:
    B^2 / (4 pi n_i m_i c^2) = B_sim^2 / (n_e |rqm_i|).
"""

import numpy as np

GAMMA_DEFAULT = 5.0 / 3.0  # adiabatic index for the ion sound speed


def ion_skin_depth(abs_rqm_i: float, n_e: float = 1.0) -> float:
    """Ion inertial length d_i = c/omega_pi  [c/omega_pe].

    Sim ions carry unit charge (n_i = n_e) with macroparticle mass |rqm_i| m_e,
    so omega_pi^2/omega_pe^2 = n_e/|rqm_i| and

        d_i = sqrt(|rqm_i| / n_e)

    At the reference density (n_e = 1) this is sqrt(|rqm_i|).  Returns nan for
    non-positive density.
    """
    if n_e <= 0.0:
        return float("nan")
    return float(np.sqrt(abs_rqm_i / n_e))


def ion_gyroperiod(abs_rqm_i: float, B_mag: float) -> float:
    """Ion gyroperiod T_ci = 2*pi*|rqm_i|/|B'|  [1/omega_pe].

    In OSIRIS units the electron cyclotron frequency is omega_ce' = B' (the
    normalised field), so the ion cyclotron frequency is omega_ci' = B'/|rqm_i|
    (rqm = m/q folds in the charge state), giving the gyroperiod

        T_ci = 2*pi / omega_ci' = 2*pi*|rqm_i| / |B'|.

    ``B_mag`` is the upstream field magnitude in OSIRIS units (B_0 = m_e c
    omega_pe / e).  Returns nan for non-positive |B'|.
    """
    if not (np.isfinite(B_mag) and B_mag > 0.0):
        return float("nan")
    return float(2.0 * np.pi * abs_rqm_i / B_mag)


def compute_dimensionless(prim: dict, v_shock: float, abs_rqm_i: float,
                          gamma: float = GAMMA_DEFAULT) -> dict:
    """Dimensionless parameters from region-averaged primitives (OSIRIS units).

    Parameters
    ----------
    prim : dict
        Region averages with keys ``n_e`` [n_0], ``T_e``, ``T_i`` (isotropic,
        [m_e c^2]) and ``B2`` = b1^2+b2^2+b3^2 [B_0^2].
    v_shock : float
        Shock velocity [c].
    abs_rqm_i : float
        |rqm_i| = m_i/(Z_i m_e), the OSIRIS ion mass-per-charge [m_e/e]; equals
        the sim ion macroparticle mass in m_e.  Z_i is already folded in, so it
        is used directly in sigma, v_A and c_s with no extra Z_i factor.
    gamma : float
        Adiabatic index for the ion sound speed (default 5/3).

    Returns
    -------
    dict with keys beta, sigma, v_A, M_A, c_s, M_s, T_e_Ti, d_i.
    """
    n_e = prim["n_e"]
    T_e = prim["T_e"]
    T_i = prim["T_i"]
    B2 = prim["B2"]

    # plasma beta = thermal pressure / magnetic pressure
    #   P_th = n_e (T_e + T_i),  P_B = B_sim^2 / 2   [both n_0 m_e c^2]
    P_thermal = n_e * (T_e + T_i)
    P_magnetic = B2 / 2.0
    beta = P_thermal / P_magnetic

    # magnetization sigma = B^2 / (4 pi n_i m_i c^2) = B_sim^2 / (n_e |rqm_i|)
    sigma = B2 / (abs_rqm_i * n_e)

    # Alfven speed: v_A^2/c^2 = B^2/(4 pi n_i m_i)/c^2 = sigma
    v_A = np.sqrt(sigma)
    M_A = v_shock / v_A

    # ion sound speed: c_s^2 = gamma (P_e + P_i)/(n_i m_i) = gamma (T_e + T_i)/|rqm_i|
    cs2 = gamma * (T_e + T_i) / abs_rqm_i
    c_s = np.sqrt(max(cs2, 0.0))
    M_s = v_shock / c_s if c_s > 0.0 else float("nan")

    T_e_Ti = T_e / T_i if T_i > 0.0 else float("nan")
    d_i = ion_skin_depth(abs_rqm_i, n_e)

    return {
        "beta": beta, "sigma": sigma, "v_A": v_A, "M_A": M_A,
        "c_s": c_s, "M_s": M_s, "T_e_Ti": T_e_Ti, "d_i": d_i,
    }


def magnetic_reynolds(T_e_sim: float, n_e_sim: float, v_shock: float, L_sim: float,
                      norm_density, d_e, Z_i: int, ion: str = "Al") -> float:
    """Physical magnetic Reynolds number Rm = v_shock * L / eta_m.

    A PIC run is collisionless; this is the *equivalent* resistive Rm of the real
    plasma the run represents, from the Spitzer conductivity at the run's physical
    density and the measured electron temperature.  astropy/plasmapy are imported
    lazily so the rest of this module stays dependency-light for CI.

    Parameters
    ----------
    T_e_sim, n_e_sim : float
        Region-mean electron temperature [m_e c^2] and density [n_0].
    v_shock : float
        Shock velocity [c].
    L_sim : float
        System (box) size [c/omega_pe].
    norm_density : astropy.units.Quantity
        Reference density n_0 (so n_e_sim * norm_density is the physical density).
    d_e : astropy.units.Quantity
        Electron inertial length c/omega_pe in length units (so L_sim * d_e is the
        physical box size).
    Z_i : int
        Ion charge state, used for the ion in the Coulomb logarithm.
    ion : str
        Ion element symbol for plasmapy's particle string (default "Al").
    """
    if not (np.isfinite(T_e_sim) and T_e_sim > 0.0 and np.isfinite(n_e_sim) and n_e_sim > 0.0):
        return float("nan")

    import astropy.constants
    import astropy.units as u
    from plasmapy.formulary import Mag_Reynolds, Spitzer_resistivity

    m_e_c2_eV = float((astropy.constants.m_e * astropy.constants.c**2).to(u.eV).value)
    T_e = (T_e_sim * m_e_c2_eV) * u.eV
    n_e = (n_e_sim * norm_density).to(u.m**-3)
    rho_m = Spitzer_resistivity(T=T_e, n=n_e, species=("e", f"{ion} {Z_i}+"), z_mean=float(Z_i))
    sigma = (1.0 / rho_m).to(u.S / u.m)
    U = (v_shock * astropy.constants.c).to(u.m / u.s)
    L = (L_sim * d_e).to(u.m)
    return float(Mag_Reynolds(U, L, sigma))
