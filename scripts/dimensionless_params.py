"""dimensionless_params — upstream/downstream dimensionless plasma parameters.

Science
-------
For one OSIRIS dump this reports the dimensionless numbers that characterise the
shock, upstream and downstream, in OSIRIS normalised units:

    beta   = 2 n_e (T_e + T_i) / B^2          thermal / magnetic pressure
    sigma  = B^2 / (|rqm_i| n_e)              magnetization (= v_A^2)
    M_A    = v_shock / v_A,  v_A = sqrt(sigma) Alfvénic Mach number
    M_s    = v_shock / c_s,  c_s = sqrt(γ(T_e+T_i)/|rqm_i|)  sonic Mach number
    T_e/T_i                                   isotropic temperature ratio
    d_i    = sqrt(|rqm_i|/n_e)                ion skin depth [c/ωpe]
    N_d_i  = L_box / d_i                      ion skin depths across the box
    Rm     = v_shock L_box / eta_m            physical (Spitzer) magnetic Reynolds #

All formulas are owned by the pure module ``src/dimensionless_params.py``
(``compute_dimensionless``, ``ion_skin_depth``, and ``magnetic_reynolds`` for the
physical Spitzer Rm).  Temperatures are T = |rqm| · 2nd-moment [m_e c^2] (the sim
macroparticle temperature, which pairs with n_e to give n_e·T_i with no extra
Z_i).  M_A here uses the single YAML ``shock.v_shock`` (not the fitted
trajectory), so it reports the run's nominal Mach number.

Validation
----------
This file is orchestration only; it contains no physics formulas.  To validate a
number, read the named function in ``src/dimensionless_params.py`` and its test in
``tests/test_dimensionless_params.py``.  The result is saved to an inspectable
``.npz`` whose schema is the ``DimensionlessParamsResult`` dataclass below.

Usage
-----
    python scripts/dimensionless_params.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--output ...]
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
import shock_state
import dimensionless_params as dp

_PARAM_KEYS = ("beta", "sigma", "v_A", "M_A", "c_s", "M_s", "T_e_Ti", "d_i")


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class DimensionlessParamsResult:
    # --- profiles on the phase-space grid ---
    x_axis: np.ndarray            # [c/wpe]
    n_e: np.ndarray               # electron density [n_0]
    T_par_e: np.ndarray           # parallel electron T [m_e c^2]
    T_perp_e: np.ndarray          # perpendicular electron T [m_e c^2]
    T_iso_e: np.ndarray           # isotropic electron T [m_e c^2]
    T_par_i: np.ndarray           # parallel ion T [m_e c^2]
    T_perp_i: np.ndarray          # perpendicular ion T [m_e c^2]
    T_iso_i: np.ndarray           # isotropic ion T [m_e c^2]
    B2: np.ndarray                # |B|^2 [B_0^2]
    # --- system-scale scalars ---
    L_box: float                  # full simulation box [c/wpe]
    d_i_ref: float                # ion skin depth at n_e=1 [c/wpe]
    # --- metadata ---
    t_val: int                    # dump index
    t_sim: float                  # simulation time [1/wpe]
    x_shock: float                # shock position [c/wpe]
    v_shock: float                # shock velocity (YAML seed) [c]
    upstream_window_ncells: int   # upstream averaging window [cells]
    downstream_window_ncells: int # downstream averaging window [cells]
    Z_i: int                      # ion charge state
    abs_rqm_i: float              # |rqm_i| = m_i/(Z_i m_e) [m_e/e]
    config_path: str              # absolute path of the analysis config used
    # --- region groups (primitives + dimensionless params + N_d_i + Rm) ---
    up: dict                      # flattened to up_<key> in the .npz
    dn: dict                      # flattened to dn_<key> in the .npz


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in src/dimensionless_params.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> DimensionlessParamsResult:
    """Load one dump, compute the dimensionless parameters up/downstream."""
    print("Loading HDF5 files...")
    # M_A uses the tuned YAML shock.v_shock (the single source of the shock speed).
    st = shock_state.load_shock_state(cfg, timestep_idx)

    abs_rqm_i = st.abs_rqm_i
    L_box = st.L_box

    # Dimensionless parameters per region (physics: compute_dimensionless).
    params_up = dp.compute_dimensionless(st.prim_up, st.v_shock, abs_rqm_i)
    params_dn = dp.compute_dimensionless(st.prim_dn, st.v_shock, abs_rqm_i)

    # System-scale quantities. d_i_ref is the skin depth at n_e=1; N_d_i is the
    # number of skin depths across the box; Rm is the physical Spitzer value.
    d_i_ref = dp.ion_skin_depth(abs_rqm_i)
    n_di = {"up": L_box / params_up["d_i"] if np.isfinite(params_up["d_i"]) else float("nan"),
            "dn": L_box / params_dn["d_i"] if np.isfinite(params_dn["d_i"]) else float("nan")}
    rm = {side: dp.magnetic_reynolds(prim["T_e"], prim["n_e"], st.v_shock, L_box,
                                     st.sim.norm_density, st.sim.d_e(), st.Z_i)
          for side, prim in (("up", st.prim_up), ("dn", st.prim_dn))}

    # Region groups = primitives + dimensionless params + N_d_i + Rm (flattened
    # to up_<key>/dn_<key> in the .npz; no key collisions between the sets).
    up = {**st.prim_up, **params_up, "N_d_i": n_di["up"], "Rm": rm["up"]}
    dn = {**st.prim_dn, **params_dn, "N_d_i": n_di["dn"], "Rm": rm["dn"]}

    return DimensionlessParamsResult(
        x_axis=st.x_pha, n_e=st.n_e,
        T_par_e=st.T_par["e"], T_perp_e=st.T_perp["e"], T_iso_e=st.T_iso["e"],
        T_par_i=st.T_par["al"], T_perp_i=st.T_perp["al"], T_iso_i=st.T_iso["al"],
        B2=st.B2,
        L_box=float(L_box), d_i_ref=float(d_i_ref),
        t_val=int(st.t_val), t_sim=float(st.t_sim),
        x_shock=float(st.x_shock), v_shock=float(st.v_shock),
        upstream_window_ncells=int(st.up_ncells), downstream_window_ncells=int(st.dn_ncells),
        Z_i=int(st.Z_i), abs_rqm_i=float(abs_rqm_i), config_path=config_path,
        up=up, dn=dn,
    )


def _print_summary(r: DimensionlessParamsResult) -> None:
    """Console tables of primitives, dimensionless parameters, and system scales."""
    m_ratio = r.Z_i * r.abs_rqm_i        # physical mass ratio m_i/m_e = Z_i |rqm_i|
    print(f"t_sim   : {r.t_sim:.1f}  [omega_pe^-1]")
    print(f"x_shock : {r.x_shock:.1f}  [c/omega_pe]   v_shock : {r.v_shock:.4f}  [c]")
    print("\n" + "=" * 70)
    print("  Dimensionless Plasma Parameters")
    print("=" * 70)
    print(f"\n  Ion species    Al^{r.Z_i}+")
    print(f"  |rqm_i|        = {r.abs_rqm_i:.5f}  [m_e/e]  (OSIRIS m/q; = sim macroparticle mass / m_e)")
    print(f"  m_i / m_e      = Z_i * |rqm_i| = {m_ratio:.1f}  (physical ion mass ratio)")
    print(f"  v_shock        = {r.v_shock:.5f}  [c]")

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  {'Primitive quantity':<22} {'Upstream':>14} {'Downstream':>14}  Unit")
    print(sep)
    prim_rows = [
        ("n_e", "n_e", "n_0"), ("T_e  (iso)", "T_e", "m_e c^2"),
        ("T_e  (par)", "T_e_par", "m_e c^2"), ("T_e  (perp)", "T_e_perp", "m_e c^2"),
        ("T_i  (iso)", "T_i", "m_e c^2"), ("T_i  (par)", "T_i_par", "m_e c^2"),
        ("T_i  (perp)", "T_i_perp", "m_e c^2"), ("B^2", "B2", "B_0^2"),
    ]
    for name, key, unit in prim_rows:
        print(f"  {name:<22} {r.up[key]:>14.4g} {r.dn[key]:>14.4g}  {unit}")
    print(f"  {'T_e + T_i':<22} {r.up['T_e']+r.up['T_i']:>14.4g} "
          f"{r.dn['T_e']+r.dn['T_i']:>14.4g}  m_e c^2")
    print(f"  {'|B|':<22} {np.sqrt(r.up['B2']):>14.4g} {np.sqrt(r.dn['B2']):>14.4g}  B_0")

    print(f"\n{sep}")
    print(f"  {'Parameter':<22} {'Upstream':>14} {'Downstream':>14}  Formula (sim units)")
    print(sep)
    param_rows = [
        ("beta", "beta", "2 * n_e * (T_e + T_i) / B^2"),
        ("sigma", "sigma", "B^2 / (|rqm_i| * n_e)           [n_i = n_e]"),
        ("v_A  [c]", "v_A", "sqrt(sigma) = sqrt(B^2 / (|rqm_i| * n_e))"),
        ("M_A  = v_sh / v_A", "M_A", "v_shock / v_A"),
        ("c_s  [c]", "c_s", "sqrt(gamma * (T_e + T_i) / |rqm_i|)"),
        ("M_s  = v_sh / c_s", "M_s", "v_shock / c_s"),
        ("T_e / T_i", "T_e_Ti", "T_e_iso / T_i_iso"),
        ("d_i  [c/wpe]", "d_i", "sqrt(|rqm_i| / n_e)             [n_i = n_e]"),
    ]
    for name, key, formula in param_rows:
        print(f"  {name:<22} {r.up[key]:>14.4g} {r.dn[key]:>14.4g}  {formula}")

    print(f"\n{sep}")
    print(f"  {'System-scale quantity':<22} {'Upstream':>14} {'Downstream':>14}  Notes")
    print(sep)
    print(f"  {'L_box  [c/wpe]':<22} {r.L_box:>14.4g} {'':>14}  full simulation box")
    print(f"  {'d_i_ref  [c/wpe]':<22} {r.d_i_ref:>14.4g} {'':>14}  sqrt(|rqm_i|), at n_e = 1")
    print(f"  {'N_d_i  = L_box/d_i':<22} {r.up['N_d_i']:>14.4g} {r.dn['N_d_i']:>14.4g}  ion skin depths across box")
    print(f"  {'Rm (Spitzer)':<22} {r.up['Rm']:>14.4g} {r.dn['Rm']:>14.4g}  v_sh*L_box/eta_m (physical)")
    print()


# ---------------------------------------------------------------------------
# CLI  (no plot — this analysis is a parameter table)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute upstream/downstream dimensionless plasma parameters.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/dimensionless_params_t{t:06d}.npz).")
    plot_style.add_publication_arg(parser)
    # --units accepted for a uniform suite CLI, but this script emits a parameter
    # table (no figure), so it is a no-op here; d_i is already reported in native units.
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][args.timestep_idx]
    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(cfg['times'])} dumps)")

    result = compute(cfg, args.timestep_idx, config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, sim_dir, "dimensionless_params", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
