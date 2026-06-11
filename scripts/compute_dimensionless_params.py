"""Compute dimensionless plasma parameters upstream and downstream of an OSIRIS shock.

Parameters reported
-------------------
  beta     — plasma beta: thermal pressure / magnetic pressure
               beta = 2 * n_e * (T_e + T_i) / B^2

  sigma    — magnetization: magnetic energy / ion rest-mass energy density
               sigma = B^2 / (|rqm_i| * n_e)

  M_A      — Alfvenic Mach number: v_shock / v_A
               v_A [c] = sqrt( B^2 / (|rqm_i| * n_e) )   (= sqrt(sigma))

  M_s      — sonic Mach number: v_shock / c_s
               c_s [c] = sqrt( gamma * (T_e + T_i) / |rqm_i| )

  T_e/T_i  — electron-to-ion temperature ratio (isotropic T = (T_par + 2 T_perp) / 3)

  d_i      — ion inertial length (skin depth) [c/omega_pe]
               d_i = sqrt(|rqm_i| / n_e)   (sim ions carry unit charge => n_i = n_e)
               At reference density (n_e = 1) this is sqrt(|rqm_i|).
               N_d_i = L_box / d_i  = number of ion skin depths across the system.

  Rm       — magnetic Reynolds number (physical): Rm = v_shock * L_box / eta_m,
               eta_m = 1 / (mu_0 * sigma_Spitzer).  sigma is the Spitzer conductivity
               from physical T_e, n_e, Z_i and the Coulomb logarithm (via plasmapy).
               PIC is collisionless, so this is the *equivalent* resistive Rm of the
               real plasma the run represents, not an intrinsic simulation quantity.

Temperatures are computed as T = |rqm| * second_central_moment(p1 or p2) [m_e c^2],
which is the standard OSIRIS convention.  This T is the *simulation macroparticle*
temperature (macroparticle mass = |rqm_i| * m_e).  It pairs with the electron
density n_e (= sim ion density, since sim ions carry unit charge) to give the
correct ion pressure n_e * T_i without any extra Z_i factor.

OSIRIS normalisation
--------------------
  lengths     c / omega_pe
  velocities  c
  B fields    B_0 = m_e c omega_pe / e
  densities   n_0
  energies    m_e c^2

Crucial identity (Gaussian):
    B_0^2 = 4 pi n_0 m_e c^2
  => magnetic pressure in sim units:  P_B  = B_sim^2 / 2         [n_0 m_e c^2]
  => thermal pressure in sim units:   P_th = n_e * (T_e + T_i)   [n_0 m_e c^2]

OSIRIS rqm = m/q (mass-per-charge), in units of m_e/e.  The charge state Z_i is
folded into rqm, so each sim ion macroparticle carries UNIT charge and mass
|rqm_i| * m_e:
    |rqm_i| = m_i / (Z_i m_e)   =>   physical mass ratio m_i/m_e = Z_i * |rqm_i|

Sim ions carry unit charge, so quasineutrality is simply n_i = n_e.  Using
macroparticle mass |rqm_i| * m_e:
    B^2 / (4 pi n_i m_i c^2) = B^2 / (n_i * |rqm_i|) = B^2 / (n_e * |rqm_i|)

Averaging windows (from config, in grid cells, measured from x_shock):
    upstream   : (x_shock,                    x_shock + upstream_window_ncells * dx]
    downstream : [x_shock - downstream_window_ncells * dx,  x_shock)

Usage
-----
    python scripts/compute_dimensionless_params.py \\
        --config config/perlmutter_1.3.1d.yaml \\
        [--timestep-idx -1] \\
        [--output results/perlmutter_1.3.1d/dimensionless_params_t000360.npz]
"""

import argparse
import os
import sys

import astropy.constants
import astropy.units as u
import numpy as np
import osh5io
from plasmapy.formulary import Mag_Reynolds, Spitzer_resistivity

# electron rest energy [eV]; converts T [m_e c^2] -> T [eV]
M_E_C2_EV = float((astropy.constants.m_e * astropy.constants.c**2).to(u.eV).value)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
from analysis_utils import axis_values
import moments as mom_module
import temperature_anisotropy as ta


def region_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    """Nanmean over mask; returns nan if mask selects no cells."""
    return float(np.nanmean(arr[mask])) if mask.any() else float("nan")


def compute_dimensionless(prim: dict, v_shock: float, abs_rqm_i: float) -> dict:
    """Compute dimensionless parameters from region-averaged primitive quantities.

    All inputs in OSIRIS normalised units (see module docstring).

    Parameters
    ----------
    prim : dict
        Keys: n_e, T_e, T_i, B2  (region averages; T in m_e c^2, B^2 in B_0^2)
    v_shock : float
        Shock velocity [c], from config.
    abs_rqm_i : float
        |rqm_i| = m_i/(Z_i m_e), the OSIRIS mass-per-charge for ions [m_e/e].
        Equals the sim ion macroparticle mass in units of m_e.  Z_i is already
        folded in, so it is used directly in sigma, v_A, and c_s with no extra
        Z_i correction.
    """
    n_e = prim["n_e"]  # electron number density [n_0]
    T_e = prim["T_e"]  # isotropic electron temperature = |rqm_e| * moment_2 [m_e c^2]
    T_i = prim["T_i"]  # isotropic ion temperature     = |rqm_i| * moment_2 [m_e c^2]
    B2  = prim["B2"]   # B_x^2 + B_y^2 + B_z^2 [B_0^2]

    # ------------------------------------------------------------------
    # plasma beta  =  thermal pressure / magnetic pressure
    #
    #   P_thermal  = n_e * (T_e + T_i)   [n_0 m_e c^2]
    #   P_magnetic = B_sim^2 / 2         [n_0 m_e c^2]  (from B_0^2 = 4 pi n_0 m_e c^2)
    # ------------------------------------------------------------------
    P_thermal  = n_e * (T_e + T_i)
    P_magnetic = B2 / 2.0
    beta = P_thermal / P_magnetic

    # ------------------------------------------------------------------
    # magnetization  sigma  =  B^2 / (4 pi n_i m_i c^2)
    #
    # Sim ions carry unit charge (n_i = n_e) with macroparticle mass
    # |rqm_i| * m_e, and B_0^2 = 4 pi n_0 m_e c^2:
    #   sigma = B_sim^2 * B_0^2 / (4 pi n_i n_0 m_i c^2)
    #         = B_sim^2 * (4 pi n_0 m_e c^2) / (4 pi n_0 * n_i * |rqm_i| m_e * c^2)
    #         = B_sim^2 / (n_i * |rqm_i|)
    #         = B_sim^2 / (n_e * |rqm_i|)
    # ------------------------------------------------------------------
    sigma = B2 / (abs_rqm_i * n_e)

    # ------------------------------------------------------------------
    # Alfven speed  v_A  [units of c]
    #
    # v_A^2 / c^2 = B^2 / (4 pi n_i m_i) / c^2 = sigma   (same substitution)
    # ------------------------------------------------------------------
    v_A = np.sqrt(sigma)   # sigma = v_A^2 always in these units
    M_A = v_shock / v_A

    # ------------------------------------------------------------------
    # ion sound speed  c_s  [units of c]
    #
    # c_s^2 = gamma * (P_e + P_i) / (n_i m_i)
    #       = gamma * (n_e T_e + n_i T_i) / (n_i * |rqm_i| m_e)
    #       = gamma * (T_e + T_i) / |rqm_i|        [n_i = n_e, sim mass = |rqm_i| m_e]
    # ------------------------------------------------------------------
    GAMMA = 5.0 / 3.0
    cs2 = GAMMA * (T_e + T_i) / abs_rqm_i
    c_s = np.sqrt(max(cs2, 0.0))
    M_s = v_shock / c_s if c_s > 0.0 else float("nan")

    # ------------------------------------------------------------------
    # electron-to-ion temperature ratio
    # ------------------------------------------------------------------
    T_e_Ti = T_e / T_i if T_i > 0.0 else float("nan")

    # ------------------------------------------------------------------
    # ion inertial length (skin depth)  d_i  [c/omega_pe]
    #
    # d_i = c/omega_pi.  Sim ions carry unit charge (n_i = n_e) with
    # macroparticle mass |rqm_i| m_e, so omega_pi^2/omega_pe^2 = n_e/|rqm_i|:
    #   d_i = sqrt(|rqm_i| / n_e)   (= sqrt(|rqm_i|) at reference density n_e = 1)
    # ------------------------------------------------------------------
    d_i = np.sqrt(abs_rqm_i / n_e) if n_e > 0.0 else float("nan")

    return {
        "beta":   beta,
        "sigma":  sigma,
        "v_A":    v_A,
        "M_A":    M_A,
        "c_s":    c_s,
        "M_s":    M_s,
        "T_e_Ti": T_e_Ti,
        "d_i":    d_i,
    }


def magnetic_reynolds(T_e_sim, n_e_sim, v_shock, L_sim, sim, Z_i):
    """Physical magnetic Reynolds number Rm = v_shock * L / eta_m.

    A PIC run is collisionless; this is the *equivalent* resistive Rm of the
    real plasma the run represents, obtained from the Spitzer conductivity at
    the run's physical density and the measured electron temperature.

    Parameters
    ----------
    T_e_sim : float
        Region-mean electron temperature [m_e c^2].
    n_e_sim : float
        Region-mean electron density [n_0].
    v_shock : float
        Shock velocity [c].
    L_sim : float
        System (box) size [c/omega_pe].
    sim : analysis_utils.MagShockZRun
        Provides the reference density and electron inertial length d_e [cm].
    Z_i : int
        Ion charge state (Aluminium), used for the ion in the Coulomb log.
    """
    if not (np.isfinite(T_e_sim) and T_e_sim > 0.0 and np.isfinite(n_e_sim) and n_e_sim > 0.0):
        return float("nan")
    T_e   = (T_e_sim * M_E_C2_EV) * u.eV               # physical electron temperature
    n_e   = (n_e_sim * sim.norm_density).to(u.m**-3)   # physical electron density
    rho_m = Spitzer_resistivity(T=T_e, n=n_e, species=("e", f"Al {Z_i}+"), z_mean=float(Z_i))
    sigma = (1.0 / rho_m).to(u.S / u.m)                # Spitzer conductivity
    U     = (v_shock * astropy.constants.c).to(u.m / u.s)
    L     = (L_sim * sim.d_e()).to(u.m)                # d_e() returns cm
    return float(Mag_Reynolds(U, L, sigma))


def main():
    parser = argparse.ArgumentParser(
        description="Compute upstream/downstream dimensionless plasma parameters."
    )
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument(
        "--timestep-idx",
        type=int,
        default=-1,
        dest="timestep_idx",
        help="Index into config times list (default: -1, last dump).",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][args.timestep_idx]

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(cfg['times'])} dumps)")

    sim = analysis_utils.run_from_config(cfg)
    spec = analysis_utils.RunSpec.from_sim_dir(sim_dir)

    # Ion charge state and simulation (reduced) mass ratio
    Z_i       = spec.charge_state("al")    # number of stripped electrons (run spec)
    abs_rqm_i = abs(sim.rqm)                # |rqm_i| = m/q = m_i/(Z_i m_e) [m_e/e]
    m_ratio   = Z_i * abs_rqm_i             # physical mass ratio m_i / m_e = Z_i * |rqm_i|

    # Averaging window (grid cells from x_shock, into each side)
    up_ncells = int(cfg.get("upstream_window_ncells",   200))
    dn_ncells = int(cfg.get("downstream_window_ncells", 200))

    species = ["al", "e"]
    species_rqm = {sp: sim.rqm_of(sp) for sp in species}  # per-species rqm from deck

    print("\nLoading HDF5 files...")
    pha_p1 = {
        sp: osh5io.read_h5(analysis_utils.phase_path(sim_dir, "p1x1", sp, t_val))
        for sp in species
    }
    pha_p2 = {
        sp: osh5io.read_h5(analysis_utils.phase_path(sim_dir, "p2x1", sp, t_val))
        for sp in species
    }
    fld = {
        name: osh5io.read_h5(analysis_utils.field_path(sim_dir, name, t_val))
        for name in ["b1", "b2", "b3"]
    }

    x_pha = axis_values(pha_p1["al"], ax_idx=1)  # phase-space spatial grid [c/omega_pe]
    x_fld = axis_values(fld["b1"],   ax_idx=0)   # field spatial grid        [c/omega_pe]
    t_sim = float(pha_p1["al"].run_attrs["TIME"][0])

    dump    = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    v_shock = dump["v_shock"]   # [c]
    x_shock = dump["x_shock"]  # [c/omega_pe]

    print(f"t_sim   : {t_sim:.1f}  [omega_pe^-1]")
    print(f"x_shock : {x_shock:.1f}  [c/omega_pe]")
    print(f"v_shock : {v_shock:.4f}  [c]")

    # ------------------------------------------------------------------
    # Temperature profiles  T(x)  [m_e c^2]
    #
    # T = |rqm| * <(p - <p>)^2>
    #   p is in OSIRIS normalised momentum units (m_e c = 1).
    #   |rqm| scales the variance from electron-mass units to species mass:
    #     electrons:  |rqm_e| = 1      =>  T_e in m_e c^2
    #     Al ions:    |rqm_i|           =>  T_i in m_e c^2  (sim reduced mass)
    # T_iso = (T_par + 2 T_perp) / 3   [m_e c^2]
    # ------------------------------------------------------------------
    T_par  = {sp: ta.temperature_profile(pha_p1[sp], species_rqm[sp], "p1") for sp in species}
    T_perp = {sp: ta.temperature_profile(pha_p2[sp], species_rqm[sp], "p2") for sp in species}
    T_iso  = {sp: (T_par[sp] + 2.0 * T_perp[sp]) / 3.0 for sp in species}

    # ------------------------------------------------------------------
    # Electron number density profile  n_e(x)  [n_0]
    #
    # Zeroth moment of f(p1, x1) over p1.  Ion density is not loaded
    # separately; sim ions carry unit charge, so n_i = n_e is used
    # wherever n_i would appear (see compute_dimensionless).
    # ------------------------------------------------------------------
    n_e = np.abs(mom_module.moment(pha_p1["e"], axis="p1", order=0))

    # ------------------------------------------------------------------
    # Magnetic field squared  B^2(x)  [B_0^2]
    #
    # Time-averaged (savg) fields; B^2 = b1^2 + b2^2 + b3^2.
    # Interpolate from the field grid onto the phase-space grid.
    # ------------------------------------------------------------------
    b1 = np.asarray(fld["b1"])
    b2 = np.asarray(fld["b2"])
    b3 = np.asarray(fld["b3"])
    B2_fld = b1**2 + b2**2 + b3**2            # on field grid
    B2     = np.interp(x_pha, x_fld, B2_fld)  # on phase-space grid

    # ------------------------------------------------------------------
    # Averaging windows (grid cells from x_shock)
    #
    #   upstream   : undisturbed plasma just to the right of the shock
    #   downstream : compressed plasma just to the left of the shock
    # ------------------------------------------------------------------
    dx = float(x_pha[1] - x_pha[0])
    upstream_mask, downstream_mask = analysis_utils.window_masks(
        x_pha, x_shock, dx, up_ncells, dn_ncells
    )

    print(f"\nUpstream   window: x in ({x_shock:.1f}, {x_shock + up_ncells * dx:.1f}]  "
          f"({up_ncells} cells, dx={dx:.2f})")
    print(f"Downstream window: x in [{x_shock - dn_ncells * dx:.1f}, {x_shock:.1f})  "
          f"({dn_ncells} cells, dx={dx:.2f})")

    # ------------------------------------------------------------------
    # Region-averaged primitive quantities
    # ------------------------------------------------------------------
    def region_primitives(mask: np.ndarray) -> dict:
        return {
            "n_e":      region_mean(n_e,             mask),  # [n_0]
            "T_e":      region_mean(T_iso["e"],      mask),  # isotropic [m_e c^2]
            "T_e_par":  region_mean(T_par["e"],      mask),
            "T_e_perp": region_mean(T_perp["e"],     mask),
            "T_i":      region_mean(T_iso["al"],     mask),
            "T_i_par":  region_mean(T_par["al"],     mask),
            "T_i_perp": region_mean(T_perp["al"],    mask),
            "B2":       region_mean(B2,              mask),  # [B_0^2]
        }

    prim_up = region_primitives(upstream_mask)
    prim_dn = region_primitives(downstream_mask)

    # ------------------------------------------------------------------
    # Dimensionless parameters
    # ------------------------------------------------------------------
    params_up = compute_dimensionless(prim_up, v_shock, abs_rqm_i)
    params_dn = compute_dimensionless(prim_dn, v_shock, abs_rqm_i)

    # ------------------------------------------------------------------
    # System-scale quantities
    #
    #   L_box   : full simulation box size [c/omega_pe]
    #   N_d_i   : number of ion skin depths across the box (per-region density)
    #   Rm      : physical magnetic Reynolds number (Spitzer; per-region T_e, n_e)
    # ------------------------------------------------------------------
    x_box_lo = float(fld["b1"].axes[0].min)
    x_box_hi = float(fld["b1"].axes[0].max)
    L_box    = x_box_hi - x_box_lo                       # [c/omega_pe]
    d_i_ref  = float(np.sqrt(abs_rqm_i))                 # ion skin depth at n_e = 1

    N_di_up = L_box / params_up["d_i"] if np.isfinite(params_up["d_i"]) else float("nan")
    N_di_dn = L_box / params_dn["d_i"] if np.isfinite(params_dn["d_i"]) else float("nan")

    Rm_up = magnetic_reynolds(prim_up["T_e"], prim_up["n_e"], v_shock, L_box, sim, Z_i)
    Rm_dn = magnetic_reynolds(prim_dn["T_e"], prim_dn["n_e"], v_shock, L_box, sim, Z_i)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Dimensionless Plasma Parameters")
    print("=" * 70)

    print(f"\n  Ion species    Al^{Z_i}+")
    print(f"  |rqm_i|        = {abs_rqm_i:.5f}  [m_e/e]  (OSIRIS m/q; = sim macroparticle mass / m_e)")
    print(f"  m_i / m_e      = Z_i * |rqm_i| = {m_ratio:.1f}  (physical ion mass ratio)")
    print(f"  v_shock        = {v_shock:.5f}  [c]")
    print(f"  gamma          = {5/3:.4f}  (adiabatic index for c_s)")

    sep = "-" * 70

    print(f"\n{sep}")
    print(f"  {'Primitive quantity':<22} {'Upstream':>14} {'Downstream':>14}  Unit")
    print(sep)
    prim_rows = [
        ("n_e",          prim_up["n_e"],       prim_dn["n_e"],       "n_0"),
        ("T_e  (iso)",   prim_up["T_e"],        prim_dn["T_e"],        "m_e c^2"),
        ("T_e  (par)",   prim_up["T_e_par"],    prim_dn["T_e_par"],    "m_e c^2"),
        ("T_e  (perp)",  prim_up["T_e_perp"],   prim_dn["T_e_perp"],   "m_e c^2"),
        ("T_i  (iso)",   prim_up["T_i"],        prim_dn["T_i"],        "m_e c^2"),
        ("T_i  (par)",   prim_up["T_i_par"],    prim_dn["T_i_par"],    "m_e c^2"),
        ("T_i  (perp)",  prim_up["T_i_perp"],   prim_dn["T_i_perp"],   "m_e c^2"),
        ("T_e + T_i",
         prim_up["T_e"] + prim_up["T_i"],
         prim_dn["T_e"] + prim_dn["T_i"],
         "m_e c^2"),
        ("B^2",          prim_up["B2"],         prim_dn["B2"],         "B_0^2"),
        ("|B|",  np.sqrt(prim_up["B2"]), np.sqrt(prim_dn["B2"]),      "B_0"),
    ]
    for name, uval, dval, unit in prim_rows:
        print(f"  {name:<22} {uval:>14.4g} {dval:>14.4g}  {unit}")

    print(f"\n{sep}")
    print(f"  {'Parameter':<22} {'Upstream':>14} {'Downstream':>14}  Formula (sim units)")
    print(sep)
    param_rows = [
        ("beta",              "2 * n_e * (T_e + T_i) / B^2"),
        ("sigma",             "B^2 / (|rqm_i| * n_e)           [n_i = n_e]"),
        ("v_A  [c]",          "sqrt(sigma) = sqrt(B^2 / (|rqm_i| * n_e))"),
        ("M_A  = v_sh / v_A", "v_shock / v_A"),
        ("c_s  [c]",          "sqrt(gamma * (T_e + T_i) / |rqm_i|)"),
        ("M_s  = v_sh / c_s", "v_shock / c_s"),
        ("T_e / T_i",         "T_e_iso / T_i_iso"),
        ("d_i  [c/wpe]",      "sqrt(|rqm_i| / n_e)             [n_i = n_e]"),
    ]
    keys = ["beta", "sigma", "v_A", "M_A", "c_s", "M_s", "T_e_Ti", "d_i"]
    for (name, formula), key in zip(param_rows, keys):
        uval = params_up[key]
        dval = params_dn[key]
        print(f"  {name:<22} {uval:>14.4g} {dval:>14.4g}  {formula}")

    print(f"\n{sep}")
    print(f"  {'System-scale quantity':<22} {'Upstream':>14} {'Downstream':>14}  Notes")
    print(sep)
    print(f"  {'L_box  [c/wpe]':<22} {L_box:>14.4g} {'':>14}  full simulation box")
    print(f"  {'d_i_ref  [c/wpe]':<22} {d_i_ref:>14.4g} {'':>14}  sqrt(|rqm_i|), at n_e = 1")
    print(f"  {'N_d_i  = L_box/d_i':<22} {N_di_up:>14.4g} {N_di_dn:>14.4g}  ion skin depths across box")
    print(f"  {'Rm (Spitzer)':<22} {Rm_up:>14.4g} {Rm_dn:>14.4g}  v_sh*L_box/eta_m (physical)")

    print()

    # ------------------------------------------------------------------
    # Save to .npz
    # ------------------------------------------------------------------
    out_path = analysis_utils.default_output_path(
        args.output, sim_dir, "dimensionless_params", t_val
    )

    save_dict = {
        # full spatial profiles (on phase-space grid)
        "x_axis":   x_pha,
        "n_e":      n_e,
        "T_par_e":  T_par["e"],
        "T_perp_e": T_perp["e"],
        "T_iso_e":  T_iso["e"],
        "T_par_i":  T_par["al"],
        "T_perp_i": T_perp["al"],
        "T_iso_i":  T_iso["al"],
        "B2":       B2,
        # region-averaged primitives
        **{f"up_{k}": np.asarray(v) for k, v in prim_up.items()},
        **{f"dn_{k}": np.asarray(v) for k, v in prim_dn.items()},
        # dimensionless parameters
        **{f"up_{k}": np.asarray(v) for k, v in params_up.items()},
        **{f"dn_{k}": np.asarray(v) for k, v in params_dn.items()},
        # system-scale quantities
        "L_box":   np.asarray(L_box),
        "d_i_ref": np.asarray(d_i_ref),
        "up_N_d_i": np.asarray(N_di_up),
        "dn_N_d_i": np.asarray(N_di_dn),
        "up_Rm":    np.asarray(Rm_up),
        "dn_Rm":    np.asarray(Rm_dn),
        # scalars / metadata
        "t_val":                    np.asarray(t_val),
        "t_sim":                    np.asarray(t_sim),
        "x_shock":                  np.asarray(x_shock),
        "v_shock":                  np.asarray(v_shock),
        "upstream_window_ncells":   np.asarray(up_ncells),
        "downstream_window_ncells": np.asarray(dn_ncells),
        "Z_i":                      np.asarray(Z_i),
        "abs_rqm_i":                np.asarray(abs_rqm_i),
        "config_path":              np.asarray(os.path.abspath(args.config)),
    }
    np.savez(out_path, **save_dict)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
